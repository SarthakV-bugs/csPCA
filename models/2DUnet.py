import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random
import SimpleITK as sitk
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import seaborn as sns



# Mock main function if the import is not available
# def main():
#     print("Warning: Using mock data since 'preprocessing.mapping_labels' was not found.")
#     # Create a list of dummy file paths for demonstration
#     # In a real scenario, these would be paths to your .nii.gz or .mha files
#     dataset = [
#         (f'/path/to/image_{i}.nii.gz', f'/path/to/label_{i}.nii.gz') for i in range(100)
#     ]
#     return dataset


# --- Step 1: Define 2D Dataset Class ---
class T2WDataset2D(Dataset):
    """
    2D Dataset that extracts individual slices from 3D volumes.
    Much more memory efficient than 3D approach.
    """

    def __init__(self, data_list, transform=None, slice_axis=0):
        """
        Args:
            data_list: List of (image_path, label_path) tuples
            transform: Optional transform to apply
            slice_axis: Which axis to slice (0=axial, 1=sagittal, 2=coronal)
        """
        self.slice_data = []
        self.transform = transform
        self.slice_axis = slice_axis

        # Pre-load all slice references
        print("Loading dataset and extracting slice indices...")
        for img_path, label_path in tqdm(data_list):
            try:
                # Read the 3D volume to get number of slices
                img_sitk = sitk.ReadImage(img_path)
                img_np = sitk.GetArrayFromImage(img_sitk).astype('float32')

                # Store reference to file and slice index
                num_slices = img_np.shape[self.slice_axis]
                for slice_idx in range(num_slices):
                    self.slice_data.append((img_path, label_path, slice_idx))

            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue

        print(f"Total 2D slices: {len(self.slice_data)}")

    def __len__(self):
        return len(self.slice_data)

    def __getitem__(self, idx):
        img_path, label_path, slice_idx = self.slice_data[idx]

        try:
            # Load 3D volume
            img_np = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype('float32')
            label_np = sitk.GetArrayFromImage(sitk.ReadImage(label_path)).astype('int')

            # Extract 2D slice
            if self.slice_axis == 0:  # Axial
                img_slice = img_np[slice_idx, :, :]
                label_slice = label_np[slice_idx, :, :]
            elif self.slice_axis == 1:  # Sagittal
                img_slice = img_np[:, slice_idx, :]
                label_slice = label_np[:, slice_idx, :]
            else:  # Coronal
                img_slice = img_np[:, :, slice_idx]
                label_slice = label_np[:, :, slice_idx]

        except Exception as e:
            print(f"Error reading slice: {e}")
            # Return dummy data
            img_slice = np.random.rand(256, 256).astype('float32') * 255
            label_slice = (np.random.rand(256, 256) > 0.5).astype('int')

        # Convert to tensors
        img = torch.from_numpy(img_slice)
        label = torch.from_numpy(label_slice)

        # Binarize label
        label = (label > 0).long()

        # Apply transforms
        if self.transform:
            img, label = self.transform(img, label)

        # Add channel dimension to image
        return img.unsqueeze(0), label


# --- Transform for 2D ---
class Resize2DTransform:
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size

    def __call__(self, image, label):
        # Add batch and channel dims: (H, W) -> (1, 1, H, W)
        image = image.unsqueeze(0).unsqueeze(0)
        label = label.unsqueeze(0).unsqueeze(0).float()

        # Resize with interpolation
        resized_image = F.interpolate(image, size=self.target_size, mode='bilinear', align_corners=False)
        resized_label = F.interpolate(label, size=self.target_size, mode='nearest')

        # Remove batch and channel dims
        return resized_image.squeeze(0).squeeze(0), resized_label.squeeze(0).squeeze(0).long()


# --- Step 2: Define 2D U-Net Architecture ---
class DoubleConv2D(nn.Module):
    """(Conv2D -> BN -> ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(DoubleConv2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet2D(nn.Module):
    """
    Standard 2D U-Net architecture.
    20x less memory than 3D U-Net.
    """

    def __init__(self, in_ch=1, out_ch=1):
        super(UNet2D, self).__init__()

        # Encoder
        self.inc = DoubleConv2D(in_ch, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv2D(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv2D(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv2D(256, 512))

        # Bottleneck
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv2D(512, 1024))

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv2D(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv2D(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv2D(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv2D(128, 64)

        # Output
        self.outc = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up4(x)

        x = self.outc(x)
        x = torch.sigmoid(x)
        return x


# --- Step 3: Data Loading ---
from preprocessing.mapping_labels import main

dataset = main()
print(f"Dataset size: {len(dataset)}")

# Use smaller fraction for testing
dataset = dataset[:int(len(dataset) * 0.1)]

random.shuffle(dataset)
train = dataset[:int(0.8 * len(dataset))]
val = dataset[int(0.8 * len(dataset)):int(0.9 * len(dataset))]
test = dataset[int(0.9 * len(dataset)):]

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

# Create 2D datasets
resize_transform = Resize2DTransform(target_size=(256, 256))

train_dataset = T2WDataset2D(train, transform=resize_transform, slice_axis=0)
val_dataset = T2WDataset2D(val, transform=resize_transform, slice_axis=0)
test_dataset = T2WDataset2D(test, transform=resize_transform, slice_axis=0)

# DataLoaders - can use larger batch size with 2D!
batch_size = 8  # Much larger than 3D
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

print("\nChecking batch shape...")
for img, label in train_loader:
    print(f"Image batch: {img.shape}")  # [B, 1, 256, 256]
    print(f"Label batch: {label.shape}")  # [B, 256, 256]
    break

# --- Step 4: Model Setup ---
model = UNet2D(in_ch=1, out_ch=1)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)


# --- Step 5: Training Functions ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


# --- Step 6: Training Loop ---
num_epochs = 10

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate_one_epoch(model, val_loader, criterion, device)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

print("Training complete!")


# --- Step 7: Evaluation Functions ---
def calculate_dice_coefficient(pred, target, smooth=1e-8):
    intersection = (pred * target).sum().item()
    dice = (2. * intersection + smooth) / (pred.sum().item() + target.sum().item() + smooth)
    return dice


def calculate_iou(pred, target, smooth=1e-8):
    intersection = (pred * target).sum().item()
    union = pred.sum().item() + target.sum().item() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


def test_model_comprehensive(model, test_loader, device, threshold=0.5):
    model.eval()

    all_predictions = []
    all_targets = []
    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = (outputs > threshold).float().squeeze(1)

            # Collect for global metrics
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_targets.extend(labels.cpu().numpy().flatten())

            # Calculate per-image dice and iou
            for i in range(predictions.shape[0]):
                dice = calculate_dice_coefficient(predictions[i], labels[i].float())
                iou = calculate_iou(predictions[i], labels[i].float())
                dice_scores.append(dice)
                iou_scores.append(iou)

    # Convert to 0,1 ints for scikit metrics
    all_predictions = np.array(all_predictions).round().astype(int)
    all_targets = np.array(all_targets).round().astype(int)

    accuracy = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions, average='binary')
    precision = precision_score(all_targets, all_predictions, average='binary')
    recall = recall_score(all_targets, all_predictions, average='binary')
    confmat = confusion_matrix(all_targets, all_predictions)

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'dice_coefficient': np.mean(dice_scores),
        'iou_score': np.mean(iou_scores),
        'confusion_matrix': confmat
    }


def print_test_results(test_results):
    print("\n" + "=" * 50)
    print("         MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy        : {test_results['accuracy']:.4f}")
    print(f"F1 Score        : {test_results['f1_score']:.4f}")
    print(f"Precision       : {test_results['precision']:.4f}")
    print(f"Recall          : {test_results['recall']:.4f}")
    print(f"Dice Coefficient: {test_results['dice_coefficient']:.4f}")
    print(f"IoU Score       : {test_results['iou_score']:.4f}")
    print("=" * 50)
    print("Confusion Matrix:\n", test_results['confusion_matrix'])


# --- Step 8: Visualization ---
# --- Step 8: Visualization ---
def visualize_2d_prediction(model, dataset, device, sample_idx=0):
    model.eval()

    img_tensor, label_tensor = dataset[sample_idx]
    input_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    img_cpu = img_tensor.squeeze(0).cpu().numpy()
    label_cpu = label_tensor.cpu().numpy()
    pred_cpu = (output_tensor > 0.5).squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_cpu, cmap='bone')
    axes[0].set_title('Original MRI Slice')
    axes[0].axis('off')

    axes[1].imshow(label_cpu, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(pred_cpu, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

# --- Step 9: Run Final Test/Evaluation ---
test_results = test_model_comprehensive(model, test_loader, device, threshold=0.5)
print_test_results(test_results)

# Visualize a sample (choose an index available in your test set)
visualize_2d_prediction(model, test_dataset, device, sample_idx=min(20, len(test_dataset)-1)) # e.g. 20th sample