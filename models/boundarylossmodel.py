# -*- coding: utf-8 -*-
"""csPCA_Models - Local System Version - Modularized"""

"""# Install dependencies (run in terminal/command prompt first)"""
# pip install SimpleITK
# pip install nibabel
# pip install scikit-image

"""# Import Modules"""

import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
import random
import SimpleITK as sitk
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from scipy import ndimage
from skimage import morphology
import json

"""# Configuration"""

class Config:
    """Configuration class for all paths and hyperparameters"""
    
    # IMPORTANT: UPDATE THESE PATHS TO YOUR LOCAL SYSTEM
    checkpoint_dir = 'checkpoints'
    labels_root = "data/picai_labels"
    mri_root = "data/mri_images"
    
    # Dataset parameters - UPDATED TO 40 SAMPLES
    num_positive_to_use = 20  # 20 positive
    num_negative_to_use = 20  # 20 negative
    target_size = (256, 256)
    batch_size = 4
    num_workers = 0  # Change to 2 for Mac/Linux
    
    # Training parameters
    num_epochs = 10
    learning_rate = 1e-4
    patience = 3
    
    # Loss function parameters
    alpha = 0.75
    gamma = 3
    boundary_weight = 0.5
    dice_weight = 0.25
    focal_weight = 0.25
    
    # Sampling parameters
    positive_weight = 10.0
    
    # Post-processing parameters
    min_size = 50
    max_holes = 30

config = Config()

# Create checkpoint directory
os.makedirs(config.checkpoint_dir, exist_ok=True)

print("="*60)
print("CONFIGURATION")
print("="*60)
print(f"Checkpoint directory: {config.checkpoint_dir}")
print(f"Labels root: {config.labels_root}")
print(f"MRI root: {config.mri_root}")
print(f"Target dataset size: {config.num_positive_to_use + config.num_negative_to_use}")
print("="*60)

"""# Dataset Classes"""

class T2WDataset2D(Dataset):
    """2D Dataset that extracts individual slices from 3D volumes."""
    
    def __init__(self, data_list, transform=None, slice_axis=0):
        self.slice_data = []
        self.transform = transform
        self.slice_axis = slice_axis
        
        if not data_list:
            print("Warning: Empty data list provided to dataset!")
            return
        
        print("Loading dataset and extracting slice indices...")
        for img_path, label_path, is_positive in tqdm(data_list):
            try:
                img_sitk = sitk.ReadImage(img_path)
                img_np = sitk.GetArrayFromImage(img_sitk).astype('float32')
                
                num_slices = img_np.shape[self.slice_axis]
                for slice_idx in range(num_slices):
                    self.slice_data.append((img_path, label_path, slice_idx, is_positive))
                    
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        
        print(f"Total 2D slices: {len(self.slice_data)}")
    
    def __len__(self):
        return len(self.slice_data)
    
    def __getitem__(self, idx):
        img_path, label_path, slice_idx, is_positive = self.slice_data[idx]
        
        try:
            img_np = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype('float32')
            label_np = sitk.GetArrayFromImage(sitk.ReadImage(label_path)).astype('int')
            
            # extract slice
            if self.slice_axis == 0:  # Axial
                img_slice = img_np[slice_idx, :, :]
                label_slice = label_np[slice_idx, :, :]
            elif self.slice_axis == 1:  # Sagittal
                img_slice = img_np[:, slice_idx, :]
                label_slice = label_np[:, slice_idx, :]
            else:  # Coronal
                img_slice = img_np[:, :, slice_idx]
                label_slice = label_np[:, :, slice_idx]
            
            
            # Normalize to [0, 1] range
            if img_slice.max() > img_slice.min():
                img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
            else:
                img_slice = np.zeros_like(img_slice)

        except Exception as e:
            print(f"Error reading slice: {e}")
            img_slice = np.random.rand(256, 256).astype('float32')  # ✅ 0-1
            label_slice = np.zeros((256, 256), dtype=int)  # ✅ All zeros
            is_positive = False

        
        img = torch.from_numpy(img_slice)
        label = torch.from_numpy(label_slice)
        is_positive_tensor = torch.tensor(is_positive, dtype=torch.bool)
        
        label = (label > 0).long()
        
        if self.transform:
            img, label = self.transform(img, label, is_positive)
        
        return img.unsqueeze(0), label, is_positive_tensor

class Resize2DTransform:
    """Transform for resizing 2D images and labels"""
    
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size
    
    def __call__(self, image, label, is_positive=None):
        image = image.unsqueeze(0).unsqueeze(0)
        label = label.unsqueeze(0).unsqueeze(0).float()
        
        resized_image = F.interpolate(image, size=self.target_size, mode='bilinear', align_corners=False)
        resized_label = F.interpolate(label, size=self.target_size, mode='nearest')
        
        return resized_image.squeeze(0).squeeze(0), resized_label.squeeze(0).squeeze(0).long()

"""# Model Architecture"""

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
    """Standard 2D U-Net architecture"""
    
    def __init__(self, in_ch=1, out_ch=1):
        super(UNet2D, self).__init__()
        
        # Encoder
        self.inc = DoubleConv2D(in_ch, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv2D(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv2D(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv2D(256, 512))
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
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
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
        
        return self.outc(x)

"""# Loss Function"""

class BoundaryDiceFocalLoss(nn.Module):
    """Combined Boundary + Dice + Focal Loss for class imbalance"""
    
    def __init__(self, alpha=0.75, gamma=3, boundary_weight=0.5, 
                 dice_weight=0.25, focal_weight=0.25, smooth=1e-6):
        super(BoundaryDiceFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.boundary_weight = boundary_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.smooth = smooth
    
    def get_boundary(self, mask):
        """Extract boundary from segmentation mask"""
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        
        kernel = torch.ones(1, 1, 3, 3, device=mask.device)
        padded = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        eroded = F.conv2d(padded, kernel, stride=1, padding=0)
        eroded = (eroded == 9).float()
        
        boundary = mask - eroded
        return (boundary > 0).float()
    
    def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        inputs_sigmoid = torch.sigmoid(inputs)
        
        if targets.dim() == 3:
            targets = targets.unsqueeze(1).float()
        
        # Boundary loss
        target_boundary = self.get_boundary(targets)
        pred_boundary = self.get_boundary(inputs_sigmoid)
        
        if target_boundary.sum() > 0:
            boundary_loss = F.binary_cross_entropy(
                pred_boundary, target_boundary, reduction='mean'
            )
        else:
            boundary_loss = torch.tensor(0.0, device=inputs.device)
        
        # 🔧 FIX: Dice loss per-sample, then average
        dice_scores = []
        for i in range(batch_size):
            inp = inputs_sigmoid[i].view(-1)
            tgt = targets[i].view(-1)
            intersection = (inp * tgt).sum()
            dice = 1 - (2. * intersection + self.smooth) / (
                inp.sum() + tgt.sum() + self.smooth
            )
            dice_scores.append(dice)
        dice_loss = torch.stack(dice_scores).mean()
        
        # Focal loss
        BCE = F.binary_cross_entropy(inputs_sigmoid, targets, reduction='none')
        pt = torch.exp(-BCE)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE
        focal_loss = focal_loss.mean()
        
        # 🔧 FIX: Scale down the combined loss
        total = (self.boundary_weight * boundary_loss +
                self.dice_weight * dice_loss +
                self.focal_weight * focal_loss)
        
        return total

"""# Data Loading Functions"""

def build_label_map(labels_root):
    """Build mapping from case IDs to label file paths"""
    print(f"Building label map from: {labels_root}")
    
    if not os.path.exists(labels_root):
        print(f"ERROR: Labels directory does not exist: {labels_root}")
        return {}
    
    label_map = {}
    priorities = [
        "csPCa_lesion_delineations/human_expert/Pooch25",
        "csPCa_lesion_delineations/human_expert/resampled", 
        "csPCa_lesion_delineations/human_expert/original",
        "csPCa_lesion_delineations/AI"
    ]
    
    for folder in priorities:
        path = os.path.join(labels_root, folder)
        if os.path.exists(path):
            files = glob.glob(os.path.join(path, "*.nii.gz"))
            print(f"Found {len(files)} label files in {folder}")
            for f in files:
                case_id = os.path.basename(f).split(".")[0]
                if case_id not in label_map:
                    label_map[case_id] = f
        else:
            print(f"Warning: Label folder does not exist: {path}")
    
    return label_map

def find_mri_images(mri_root, label_map):
    """Find MRI images that match the label map"""
    print(f"Searching for MRI images in: {mri_root}")
    
    if not os.path.exists(mri_root):
        print(f"ERROR: MRI directory does not exist: {mri_root}")
        return []
    
    dataset = []
    positive_cases = []
    negative_cases = []
    
    # Search in fold directories
    for fold in range(5):
        fold_dir = os.path.join(mri_root, f"fold{fold}", f"picai_public_images_fold{fold}")
        
        if not os.path.exists(fold_dir):
            print(f"Warning: Fold directory does not exist: {fold_dir}")
            continue
        
        t2w_files = glob.glob(os.path.join(fold_dir, "**", "*t2w.mha"), recursive=True)
        print(f"Found {len(t2w_files)} T2W files in fold {fold}")
        
        for t2w_path in t2w_files:
            case_id = os.path.basename(t2w_path).split("_t2w")[0]
            if case_id in label_map:
                label_path = label_map[case_id]
                try:
                    label_sitk = sitk.ReadImage(label_path)
                    label_np = sitk.GetArrayFromImage(label_sitk)
                    is_positive = np.any(label_np > 0)
                    
                    if is_positive:
                        positive_cases.append((t2w_path, label_path, True))
                    else:
                        negative_cases.append((t2w_path, label_path, False))
                        
                except Exception as e:
                    print(f"Error reading label file {label_path}: {e}")
                    continue
    
    dataset = positive_cases + negative_cases
    
    print(f"Total positive cases found: {len(positive_cases)}")
    print(f"Total negative cases found: {len(negative_cases)}")
    print(f"Final dataset size: {len(dataset)}")
    
    return dataset

def create_balanced_dataset(dataset, config):
    """Create a balanced dataset with stratified splits"""
    
    if not dataset:
        print("ERROR: No dataset provided!")
        return [], [], []
    
    positive_cases = [case for case in dataset if case[2] is True]
    negative_cases = [case for case in dataset if case[2] is False]
    
    if len(positive_cases) == 0:
        print("ERROR: No positive cases found!")
        return [], [], []
    
    if len(negative_cases) == 0:
        print("ERROR: No negative cases found!")
        return [], [], []
    
    # Adjust numbers based on availability
    num_pos = min(config.num_positive_to_use, len(positive_cases))
    num_neg = min(config.num_negative_to_use, len(negative_cases))
    
    print(f"Available: {len(positive_cases)} positive, {len(negative_cases)} negative")
    print(f"Using: {num_pos} positive, {num_neg} negative")
    
    # Set random seed for reproducibility
    random.seed(42)
    random.shuffle(positive_cases)
    random.shuffle(negative_cases)
    
    # Select balanced subsets
    selected_positive = positive_cases[:num_pos]
    selected_negative = negative_cases[:num_neg]
    
    # STRATIFIED SPLIT - Ensure positives in each split
    n_train_pos = int(0.70 * num_pos)
    n_val_pos = int(0.15 * num_pos)
    n_test_pos = num_pos - n_train_pos - n_val_pos
    
    n_train_neg = int(0.70 * num_neg)
    n_val_neg = int(0.15 * num_neg)
    n_test_neg = num_neg - n_train_neg - n_val_neg
    
    # Create stratified splits
    train = selected_positive[:n_train_pos] + selected_negative[:n_train_neg]
    val = selected_positive[n_train_pos:n_train_pos+n_val_pos] + selected_negative[n_train_neg:n_train_neg+n_val_neg]
    test = selected_positive[n_train_pos+n_val_pos:] + selected_negative[n_train_neg+n_val_neg:]
    
    # Shuffle each split
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    
    # Count positives in each split
    train_pos = sum(1 for _, _, is_pos in train if is_pos)
    val_pos = sum(1 for _, _, is_pos in val if is_pos)
    test_pos = sum(1 for _, _, is_pos in test if is_pos)
    
    total = len(train) + len(val) + len(test)
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train)} cases ({len(train)/total*100:.1f}%)")
    print(f"  Val:   {len(val)} cases ({len(val)/total*100:.1f}%)")
    print(f"  Test:  {len(test)} cases ({len(test)/total*100:.1f}%)")
    
    print(f"\nPositive cases per split:")
    print(f"  Train: {train_pos}/{len(train)} ({train_pos/len(train)*100:.1f}%)")
    print(f"  Val:   {val_pos}/{len(val)} ({val_pos/len(val)*100:.1f}%)")
    print(f"  Test:  {test_pos}/{len(test)} ({test_pos/len(test)*100:.1f}%)")
    
    return train, val, test

"""# Training and Evaluation Functions"""

def get_sample_weights(dataset, positive_weight=10.0):
    """Calculate sample weights for balanced sampling"""
    weights = []
    for _, _, _, is_positive in dataset.slice_data:
        weights.append(positive_weight if is_positive else 1.0)
    return torch.DoubleTensor(weights)

def create_data_loaders(train, val, test, config):
    """Create data loaders with weighted sampling"""
    
    if not train or not val or not test:
        print("ERROR: One or more dataset splits are empty!")
        return None, None, None, None, None, None
    
    resize_transform = Resize2DTransform(target_size=config.target_size)
    
    train_dataset = T2WDataset2D(train, transform=resize_transform, slice_axis=0)
    val_dataset = T2WDataset2D(val, transform=resize_transform, slice_axis=0)
    test_dataset = T2WDataset2D(test, transform=resize_transform, slice_axis=0)
    
    if len(train_dataset) == 0:
        print("ERROR: Train dataset is empty after processing!")
        return None, None, None, None, None, None
    
    # Create weighted sampler for training
    train_weights = get_sample_weights(train_dataset, config.positive_weight)
    train_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        sampler=train_sampler,
        num_workers=config.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers
    )
    
    print(f"Data loaders created successfully!")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    for images, labels, _ in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() 
        num_batches += 1
    
    return running_loss / num_batches

def validate_one_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() 
            num_batch += 1
    
    return running_loss / num_batches

"""# Evaluation Functions - SAMPLE-WISE METRICS"""

def calculate_dice_coefficient(pred, target, smooth=1e-8):
    """Calculate Dice coefficient for a single sample"""
    pred = pred.long()
    target = target.long()
    intersection = (pred * target).sum().item()
    dice = (2. * intersection + smooth) / (pred.sum().item() + target.sum().item() + smooth)
    return dice

def calculate_iou(pred, target, smooth=1e-8):
    """Calculate IoU for a single sample"""
    pred = pred.long()
    target = target.long()
    intersection = (pred * target).sum().item()
    union = pred.sum().item() + target.sum().item() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def post_process_predictions(predictions, min_size=50, max_holes=30):
    """Remove small false positive components and fill holes"""
    cleaned = morphology.remove_small_objects(
        predictions.astype(bool), 
        min_size=min_size
    )
    cleaned = morphology.remove_small_holes(
        cleaned, 
        area_threshold=max_holes
    )
    return cleaned.astype(int)

def test_model_comprehensive(model, test_loader, device, threshold=0.5):
    """Calculate metrics PER SAMPLE (slice-wise), not pixel-wise"""
    model.eval()
    
    # Store per-sample metrics
    sample_dice_scores = []
    sample_iou_scores = []
    sample_f1_scores = []
    sample_precision_scores = []
    sample_recall_scores = []
    sample_accuracy_scores = []
    
    # For overall confusion matrix (optional)
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels, is_positive in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            predictions = (torch.sigmoid(outputs) > threshold).float().squeeze(1)
            
            # Post-processing
            predictions_np = predictions.cpu().numpy()
            for i in range(predictions_np.shape[0]):
                predictions_np[i] = post_process_predictions(
                    predictions_np[i], 
                    min_size=config.min_size,
                    max_holes=config.max_holes
                )
            predictions = torch.from_numpy(predictions_np).to(device)
            
            # Calculate metrics PER SAMPLE (each slice separately)
            for i in range(predictions.shape[0]):
                pred_slice = predictions[i].cpu().numpy().flatten()
                label_slice = labels[i].cpu().numpy().flatten()
                
                # Sample-wise Dice and IoU
                dice = calculate_dice_coefficient(predictions[i], labels[i].float())
                iou = calculate_iou(predictions[i], labels[i].float())
                
                # Convert to binary for sklearn metrics
                pred_binary = pred_slice.astype(int)
                label_binary = label_slice.astype(int)
                
                # Calculate per-sample classification metrics
                accuracy = accuracy_score(label_binary, pred_binary)
                f1 = f1_score(label_binary, pred_binary, average='binary', zero_division=0)
                precision = precision_score(label_binary, pred_binary, average='binary', zero_division=0)
                recall = recall_score(label_binary, pred_binary, average='binary', zero_division=0)
                
                # Store per-sample scores
                sample_dice_scores.append(dice)
                sample_iou_scores.append(iou)
                sample_f1_scores.append(f1)
                sample_precision_scores.append(precision)
                sample_recall_scores.append(recall)
                sample_accuracy_scores.append(accuracy)
                
                # Collect for global confusion matrix
                all_predictions.extend(pred_binary)
                all_targets.extend(label_binary)
    
    # Calculate MEAN metrics across all samples
    mean_metrics = {
        'mean_dice': np.mean(sample_dice_scores),
        'std_dice': np.std(sample_dice_scores),
        'mean_iou': np.mean(sample_iou_scores),
        'std_iou': np.std(sample_iou_scores),
        'mean_f1': np.mean(sample_f1_scores),
        'std_f1': np.std(sample_f1_scores),
        'mean_precision': np.mean(sample_precision_scores),
        'std_precision': np.std(sample_precision_scores),
        'mean_recall': np.mean(sample_recall_scores),
        'std_recall': np.std(sample_recall_scores),
        'mean_accuracy': np.mean(sample_accuracy_scores),
        'std_accuracy': np.std(sample_accuracy_scores),
        'num_samples': len(sample_dice_scores)
    }
    
    # Overall confusion matrix
    all_predictions = np.array(all_predictions).astype(int)
    all_targets = np.array(all_targets).astype(int)
    confmat = confusion_matrix(all_targets, all_predictions)
    mean_metrics['confusion_matrix'] = confmat
    
    return mean_metrics

def print_test_results(test_results):
    """Print sample-wise test results"""
    print("\n" + "=" * 60)
    print("    SAMPLE-WISE MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Number of samples evaluated: {test_results['num_samples']}")
    print("\n--- Mean Metrics Across All Samples (Sample-Wise) ---")
    print(f"Dice Coefficient : {test_results['mean_dice']:.4f} ± {test_results['std_dice']:.4f}")
    print(f"IoU Score        : {test_results['mean_iou']:.4f} ± {test_results['std_iou']:.4f}")
    print(f"F1 Score         : {test_results['mean_f1']:.4f} ± {test_results['std_f1']:.4f}")
    print(f"Precision        : {test_results['mean_precision']:.4f} ± {test_results['std_precision']:.4f}")
    print(f"Recall           : {test_results['mean_recall']:.4f} ± {test_results['std_recall']:.4f}")
    print(f"Accuracy         : {test_results['mean_accuracy']:.4f} ± {test_results['std_accuracy']:.4f}")
    print("=" * 60)
    print("Overall Confusion Matrix (all pixels combined):")
    print(test_results['confusion_matrix'])
    print("=" * 60)

def find_optimal_threshold(model, val_loader, device):
    """Find optimal threshold by maximizing F1 score on validation set"""
    model.eval()
    all_outputs = []
    all_labels = []
    
    print("\nFinding optimal threshold on validation set...")
    with torch.no_grad():
        for images, labels, _ in tqdm(val_loader, desc="Threshold tuning"):
            images = images.to(device)
            outputs = torch.sigmoid(model(images))
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())
    
    all_outputs = torch.cat(all_outputs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    best_f1 = 0
    best_threshold = 0.5
    
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (all_outputs > thresh).astype(int)
        f1 = f1_score(all_labels.flatten(), preds.flatten(), average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    print(f"✓ Optimal threshold: {best_threshold:.2f} with F1: {best_f1:.4f}")
    return best_threshold

def visualize_predictions(model, test_dataset, device, optimal_threshold, num_samples=3):
    """Visualize predictions on test samples"""
    model.eval()
    
    print(f"\nGenerating visualizations for {num_samples} samples...")
    
    for i in range(min(num_samples, len(test_dataset))):
        sample_idx = min(i * (len(test_dataset) // num_samples), len(test_dataset) - 1)
        
        img_tensor, label_tensor, _ = test_dataset[sample_idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        img_cpu = img_tensor.squeeze(0).cpu().numpy()
        label_cpu = label_tensor.cpu().numpy()
        pred_cpu = (torch.sigmoid(output_tensor) > optimal_threshold).squeeze().cpu().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_cpu, cmap='bone')
        axes[0].set_title(f'MRI Slice (Sample {sample_idx})')
        axes[0].axis('off')
        
        axes[1].imshow(label_cpu, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        axes[2].imshow(pred_cpu, cmap='gray')
        axes[2].set_title(f'Prediction (Thresh {optimal_threshold:.2f})')
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(config.checkpoint_dir, f'prediction_sample_{sample_idx}.png')
        plt.savefig(save_path)
        print(f"Saved visualization to: {save_path}")
        plt.close()

"""# Main Execution"""

def main():
    """Main execution function"""
    
    print("\n" + "="*60)
    print("STARTING DATA LOADING")
    print("="*60)
    
    # Step 1: Build label map
    label_map = build_label_map(config.labels_root)
    if not label_map:
        print("ERROR: No labels found! Please check your labels_root path.")
        return
    
    print(f"✓ Found {len(label_map)} labels")
    
    # Step 2: Find matching MRI images
    dataset = find_mri_images(config.mri_root, label_map)
    if not dataset:
        print("ERROR: No MRI images found! Please check your mri_root path.")
        return
    
    print(f"✓ Found {len(dataset)} total cases")
    
    # Step 3: Create balanced dataset
    train, val, test = create_balanced_dataset(dataset, config)
    if not train:
        print("ERROR: Could not create training dataset!")
        return
    
    print(f"✓ Created balanced dataset splits")
    
    # Step 4: Create data loaders
    result = create_data_loaders(train, val, test, config)
    if result[0] is None:
        print("ERROR: Could not create data loaders!")
        return
    
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = result
    print(f"✓ Created data loaders")
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    # Setup model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = UNet2D(in_ch=1, out_ch=1).to(device)
    criterion = BoundaryDiceFocalLoss(
        alpha=config.alpha,
        gamma=config.gamma,
        boundary_weight=config.boundary_weight,
        dice_weight=config.dice_weight,
        focal_weight=config.focal_weight
    )
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    checkpoint_path = os.path.join(config.checkpoint_dir, "best_model.pth")
    
    for epoch in range(config.num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{config.num_epochs}] - Train: {train_loss:.4f}, Val: {val_loss:.4f}, LR: {current_lr:.6f}')
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f'✅ Best model saved!')
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f'🛑 Early stopping at epoch {epoch+1}!')
                break
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    # Load best model
    if os.path.exists(checkpoint_path):
        print(f"Loading best model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(model, val_loader, device)
    
    # Test with optimal threshold
    print(f"\n{'='*60}")
    print("TESTING WITH OPTIMAL THRESHOLD")
    print(f"{'='*60}")
    test_results = test_model_comprehensive(model, test_loader, device, threshold=optimal_threshold)
    print_test_results(test_results)
    
    # Test with standard 0.5 threshold
    print(f"\n{'='*60}")
    print("TESTING WITH STANDARD THRESHOLD 0.5")
    print(f"{'='*60}")
    test_results_standard = test_model_comprehensive(model, test_loader, device, threshold=0.5)
    print_test_results(test_results_standard)
    
    # Save results
    results_path = os.path.join(config.checkpoint_dir, "test_results.json")
    results_to_save = {
        'optimal_threshold': {
            'threshold': float(optimal_threshold),
            'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v.tolist() 
                       for k, v in test_results.items()}
        },
        'standard_threshold': {
            'threshold': 0.5,
            'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v.tolist() 
                       for k, v in test_results_standard.items()}
        }
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=4)
    
    print(f"\nResults saved to: {results_path}")
    
    # Visualize predictions
    visualize_predictions(model, test_dataset, device, optimal_threshold, num_samples=3)
    
    print(f"\n{'='*60}")
    print("ALL DONE!")
    print(f"{'='*60}")
    print(f"Model saved to: {checkpoint_path}")
    print(f"Results saved to: {results_path}")
    print(f"Visualizations saved to: {config.checkpoint_dir}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nERROR occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check:")
        print("1. Data paths are correct")
        print("2. Required packages are installed")
        print("3. Data files exist and are readable")
