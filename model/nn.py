import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random
import SimpleITK as sitk
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Assuming this script exists and functions as expected
from preprocessing.mapping_labels import main

# Mock main function if the import is not available
# def main():
#     print("Warning: Using mock data since 'preprocessing.mapping_labels' was not found.")
#     # Create a list of dummy file paths for demonstration
#     # In a real scenario, these would be paths to your .nii.gz or .mha files
#     dataset = [
#         (f'/path/to/image_{i}.nii.gz', f'/path/to/label_{i}.nii.gz') for i in range(100)
#     ]
#     return dataset


# --- Step 1: Define Dataset Class ---
# This class handles loading the image and label data from disk.
class T2WDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # NOTE: For demonstration, this part will fail without real data.
        # The logic below assumes valid paths from the 'main' function.
        try:
            img_path, label_path = self.data_list[idx]
            img_np = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype('float32')
            label_np = sitk.GetArrayFromImage(sitk.ReadImage(label_path)).astype('int')
        except Exception as e:
            # If reading from path fails, create dummy data to allow the script to run
            print(f"Could not read data file, creating dummy data. Error: {e}")
            img_np = (torch.rand(20, 300, 300) * 255).numpy().astype('float32')
            label_np = (torch.rand(20, 300, 300) > 0.5).numpy().astype('int')


        # --- FIX: Convert numpy arrays to tensors and binarize the label ---
        img = torch.from_numpy(img_np)
        label_tensor = torch.from_numpy(label_np)
        # Ensure label is binary (0 or 1) for BCELoss. Any non-zero label is treated as 1.
        label = (label_tensor > 0).long()

        # Optional transforms now operate on tensors
        if self.transform:
            img, label = self.transform(img, label)

        # Return the final tensors, adding the channel dimension for the image
        return img.unsqueeze(0), label

# This class now expects and returns Tensors.
class ResizeTransform(object):
    def __init__(self, target_size=(16, 256, 256)):
        self.target_size = target_size

    def __call__(self, image, label):
        # F.interpolate for 3D data needs a 5D tensor: (N, C, D, H, W)
        # Current image shape: (D, H, W). We add Batch and Channel dims.
        image = image.unsqueeze(0).unsqueeze(0) # Shape -> (1, 1, D, H, W)

        # Resize the image using trilinear interpolation for smoothness
        resized_image = F.interpolate(image, size=self.target_size, mode='trilinear', align_corners=False)

        # Do the same for the label: add dims and convert to float for interpolation
        label = label.unsqueeze(0).unsqueeze(0).float()
        # Resize the label using nearest-neighbor interpolation to preserve integer values
        resized_label = F.interpolate(label, size=self.target_size, mode='nearest')

        # Remove the Batch and Channel dimensions before returning
        return resized_image.squeeze(0).squeeze(0), resized_label.squeeze(0).squeeze(0).long()


# --- Step 2: Define Model Architecture ---
# The DoubleConv block is a fundamental building block of the UNet.
class DoubleConv(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# The UNet3D model is designed for 3D image segmentation.
class UNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(UNet3D, self).__init__()
        self.inc = DoubleConv(in_ch, 32)
        self.down1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(128, 256))

        self.up2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(128, 64)
        self.up0 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.conv_up0 = DoubleConv(64, 32)

        self.outc = nn.Conv3d(32, out_ch, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up2(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up2(x)

        x = self.up1(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up1(x)

        x = self.up0(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up0(x)

        x = self.outc(x)
        x = torch.sigmoid(x)  # For binary segmentation
        return x


# --- Step 3: Data Loading and Splitting ---
# Call the main function from data_mapper to get the full dataset.
dataset = main()
print("Dataset populated with length:", len(dataset))

# Take a smaller fraction of the dataset for faster training on CPU
dataset_fraction = 0.1  # Use 10% of the data
small_dataset_size = int(len(dataset) * dataset_fraction)
print(f"Training on a smaller dataset of {small_dataset_size} items.")
dataset = dataset[:small_dataset_size]

# Shuffle the dataset and split into training, validation, and testing sets
random.shuffle(dataset)
train = dataset[:int(0.8 * len(dataset))]
val = dataset[int(0.8 * len(dataset)):int(0.9 * len(dataset))]
test = dataset[int(0.9 * len(dataset)):]

print("Train length:", len(train))
print("Validation length:", len(val))
print("Test length:", len(test))

# Create a transform to ensure all images are the same size.
resize_transform = ResizeTransform(target_size=(16, 256, 256))

# Create dataset and dataloader instances
train_dataset = T2WDataset(train, transform=resize_transform)
val_dataset = T2WDataset(val, transform=resize_transform)
test_dataset = T2WDataset(test, transform=resize_transform)

batch_size = 1 # Keep this low for CPU memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Checking one batch from train_loader...")
for img, label in train_loader:
    print("Image batch shape:", img.shape)   # Expected: [batch, 1, D, H, W]
    print("Label batch shape:", label.shape) # Expected: [batch, D, H, W]
    break
print("Batch check complete.")

# --- Step 4: Model and Training Setup ---
model = UNet3D(in_ch=1, out_ch=1)
criterion = nn.BCELoss()  # Binary segmentation loss
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Determine the device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)


# --- Step 5: Training and Validation Functions ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()  # Set model to training mode
    running_loss = 0.0
    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1) # Add channel dim for loss

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1) # Add channel dim for loss

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


# --- Step 6: Training Loop ---
num_epochs = 5  # Reduced number of epochs for faster training on CPU

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate_one_epoch(model, val_loader, criterion, device)

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, '
          f'Validation Loss: {val_loss:.4f}')

print("Training complete!")

import matplotlib.pyplot as plt
import numpy as np


def visualize_prediction(model, dataset, device, sample_idx=0):
    """
    Visualizes the original image, ground truth mask, and predicted mask for a single sample.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Get a single sample from the dataset
    # The dataset returns (image, label)
    img_tensor, label_tensor = dataset[sample_idx]

    # The model expects a batch, so add a batch dimension (B, C, D, H, W)
    # Also, move the input tensor to the correct device
    input_tensor = img_tensor.unsqueeze(0).to(device)

    # Perform inference without calculating gradients
    with torch.no_grad():
        # Get the model's raw output (probabilities between 0 and 1)
        output_tensor = model(input_tensor)

    # --- Process Tensors for Visualization ---
    # Remove batch and channel dimensions and move to CPU
    # Original image: [C, D, H, W] -> [D, H, W]
    img_cpu = img_tensor.squeeze(0).cpu().numpy()

    # Ground truth label: [D, H, W]
    label_cpu = label_tensor.cpu().numpy()

    # Predicted mask:
    # 1. Apply a threshold (0.5) to get a binary mask
    # 2. Squeeze to remove batch and channel dimensions: [1, 1, D, H, W] -> [D, H, W]
    # 3. Move to CPU and convert to numpy for plotting
    pred_mask_cpu = (output_tensor > 0.5).squeeze(0).squeeze(0).cpu().numpy()

    # --- Select a Slice to Display ---
    # We will show the middle slice along the depth (D) axis
    slice_idx = img_cpu.shape[0] // 2

    img_slice = img_cpu[slice_idx, :, :]
    label_slice = label_cpu[slice_idx, :, :]
    pred_slice = pred_mask_cpu[slice_idx, :, :]

    # --- Plot the Results ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_slice, cmap='bone')
    axes[0].set_title(f'Original MRI Slice #{slice_idx}')
    axes[0].axis('off')

    axes[1].imshow(label_slice, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')

    axes[2].imshow(pred_slice, cmap='gray')
    axes[2].set_title('Predicted Mask from Model')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


# --- Call the visualization function after training ---
print("\nVisualizing a sample prediction from the test set...")
# Use the test_dataset and the trained model
visualize_prediction(model, test_dataset, device)

