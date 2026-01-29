# -*- coding: utf-8 -*-
"""csPCA Models - 2.5D Version with Spatial Context

Improvements over 2D baseline:
- Uses adjacent slices (previous, current, next) as 3-channel input
- Provides spatial context for lesion awareness
- Minimal architectural change (3-channel input instead of 1-channel)

Phase transition: 2D baseline → 2.5D with spatial context
Objective: Enable lesion-aware learning by introducing depth information
"""

"""# Install dependencies (run in terminal/command prompt first)"""
# pip install SimpleITK
# pip install nibabel
# pip install scikit-image

"""# Import Modules"""

import os
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
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

# Add project root to path to allow imports from parent directories
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import load_full_dataset, get_stratified_data_splits, get_dataloaders
from data.augmentation import MedicalImageAugmentation
from training.losses import BoundaryDiceFocalLoss

"""# Configuration"""


class Config:
    # Paths (update to your local system)
    project_root = Path(__file__).parent.parent
    checkpoint_dir = project_root / 'checkpoints_2p5d'  # Separate checkpoints for 2.5D
    labels_root = project_root / 'data' / 'picai_labels'
    mri_root = project_root / 'data' / 'mri_images'

    # Dataset - SMALL TEST RUN (2.5D version)
    # Available: ~425 positive cases, ~1075 negative cases (1500 total)
    num_positive_to_use = 10  # SMALL TEST - increase to 300 for full training
    num_negative_to_use = 25  # SMALL TEST - increase to 750 for full training
    target_size = (256, 256)
    batch_size = 4  # Smaller batch for testing
    num_workers = 0  # START WITH 0 FOR DEBUGGING - increase after fix confirmed
    
    # Data splits (applied after selecting above cases)
    train_split = 0.70  # 70% for training
    val_split = 0.15   # 15% for validation
    test_split = 0.15  # 15% for testing

    # Training
    num_epochs = 3  # SMALL TEST - increase to 20 for full training
    learning_rate = 1e-4
    weight_decay = 1e-5
    patience = 5  # Increased patience for longer training
    
    # Learning rate scheduling
    lr_scheduler = 'cosine'  # 'cosine' or 'exponential'
    warmup_epochs = 2

    # Loss weights - UPDATED FOR CLASS IMBALANCE
    # Positive class is minority: ~28% positive, ~72% negative
    alpha = 0.75  # Focal loss alpha
    gamma = 2.0   # Focal loss gamma
    
    # Component weights in combined loss
    dice_weight = 0.4
    focal_weight = 0.4
    boundary_weight = 0.2
    
    # Class weighting for BCE component
    pos_weight = 2.5  # Weight positive samples more heavily

    # Data augmentation
    enable_augmentation = True
    rotation_angle = 15  # degrees
    elastic_deformation = True
    intensity_variation = 0.1  # ±10% intensity variation

    # Sampling strategy for handling class imbalance
    use_weighted_sampling = True  # Oversample positive slices
    positive_slice_weight = 3.0  # 3x more likely to sample positive slices

    # Postprocessing
    min_size = 50
    max_holes = 30
    
    # Validation/Testing
    optimal_threshold_search = True  # Search for best threshold on val set
    threshold_range = [0.1, 0.9, 0.05]  # [min, max, step]

    # 🆕 2.5D-specific configuration
    use_adjacent_slices = True  # Use previous, current, next slices
    context_slices = 1  # Number of slices on each side (1 = prev+curr+next)


config = Config()

# Create checkpoint directory
os.makedirs(config.checkpoint_dir, exist_ok=True)

print("="*60)
print("CONFIGURATION (2.5D MODEL)")
print("="*60)
print(f"Checkpoint directory: {config.checkpoint_dir}")
print(f"Labels root: {config.labels_root}")
print(f"MRI root: {config.mri_root}")
print(f"Target dataset size: {config.num_positive_to_use + config.num_negative_to_use}")
print(f"Input channels: 3 (adjacent slices - 2.5D)")
print("="*60)

"""# Dataset Classes - 2.5D VERSION"""

class T2WDataset2p5D(Dataset):
    """2.5D Dataset that extracts individual slices WITH adjacent context.
    
    Each sample returns:
    - Image: 3-channel tensor (previous slice, current slice, next slice)
    - Label: ground truth mask for current slice
    - is_positive: case-level label
    """
    
    def __init__(self, data_list, transform=None, slice_axis=0, context_slices=1):
        self.slice_data = []
        self.transform = transform
        self.slice_axis = slice_axis
        self.context_slices = context_slices  # Number of slices on each side
        
        if not data_list:
            print("Warning: Empty data list provided to dataset!")
            return
        
        print("Loading dataset and extracting slice indices (2.5D)...")
        for img_path, label_path, is_positive in tqdm(data_list):
            try:
                img_sitk = sitk.ReadImage(img_path)
                img_np = sitk.GetArrayFromImage(img_sitk).astype('float32')
                
                num_slices = img_np.shape[self.slice_axis]
                # Skip first and last slices to have context on both sides
                for slice_idx in range(self.context_slices, num_slices - self.context_slices):
                    self.slice_data.append((img_path, label_path, slice_idx, is_positive))
                    
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        
        print(f"Total 2.5D slices (with context): {len(self.slice_data)}")
    
    def __len__(self):
        return len(self.slice_data)
    
    def __getitem__(self, idx):
        img_path, label_path, slice_idx, is_positive = self.slice_data[idx]
        
        try:
            img_np = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype('float32')
            label_np = sitk.GetArrayFromImage(sitk.ReadImage(label_path)).astype('int')
            
            # Extract slices with context (previous, current, next)
            slices_to_extract = []
            for offset in [-self.context_slices, 0, self.context_slices]:
                context_idx = slice_idx + offset
                
                if self.slice_axis == 0:  # Axial
                    img_slice = img_np[context_idx, :, :]
                elif self.slice_axis == 1:  # Sagittal
                    img_slice = img_np[:, context_idx, :]
                else:  # Coronal
                    img_slice = img_np[:, :, context_idx]
                
                # Normalize per-slice
                if img_slice.max() > img_slice.min():
                    img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
                else:
                    img_slice = np.zeros_like(img_slice)
                
                slices_to_extract.append(img_slice)
            
            # Get label from current slice
            if self.slice_axis == 0:
                label_slice = label_np[slice_idx, :, :]
            elif self.slice_axis == 1:
                label_slice = label_np[:, slice_idx, :]
            else:
                label_slice = label_np[:, :, slice_idx]

        except Exception as e:
            print(f"Error reading slice: {e}")
            # Return dummy 3-channel image
            slices_to_extract = [np.random.rand(256, 256).astype('float32') for _ in range(3)]
            label_slice = np.zeros((256, 256), dtype=int)
            is_positive = False

        # Stack slices to create 3-channel image
        img_stacked = np.stack(slices_to_extract, axis=0)  # (3, H, W)
        img = torch.from_numpy(img_stacked).float()
        label = torch.from_numpy(label_slice)
        is_positive_tensor = torch.tensor(is_positive, dtype=torch.bool)
        
        label = (label > 0).long()
        
        # Resize to consistent size BEFORE augmentation
        img = img.unsqueeze(0)  # (1, 3, H, W)
        label = label.unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
        
        img_resized = F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
        label_resized = F.interpolate(label, size=(256, 256), mode='nearest')
        
        img_resized = img_resized.squeeze(0)  # (3, 256, 256)
        label_resized = label_resized.squeeze(0).squeeze(0).long()  # (256, 256)
        
        if self.transform:
            # Transform expects single-channel, so we'll apply to each channel separately
            img_channels = []
            for c in range(img_resized.shape[0]):
                img_c, _ = self.transform(img_resized[c], label_resized)
                img_channels.append(img_c)
            img_resized = torch.stack(img_channels, dim=0)
        
        # Ensure contiguous memory layout for DataLoader batching
        img_out = img_resized.contiguous()
        label_out = label_resized.contiguous()
        
        return img_out, label_out, is_positive_tensor


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

"""# Model Architecture - 2.5D VERSION"""

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

class UNet2D_2p5D(nn.Module):
    """Standard 2D U-Net with 3-channel input (2.5D)
    
    Only change from 2D version:
    - Input channels: 3 (instead of 1)
    - Everything else remains identical
    """
    
    def __init__(self, in_ch=3, out_ch=1):  # 🆕 in_ch=3 for 2.5D
        super(UNet2D_2p5D, self).__init__()
        
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

"""# Training and Evaluation Functions (identical to 2D version)"""

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
            num_batches += 1
    
    return running_loss / num_batches

"""# Evaluation Functions"""

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
    
    sample_dice_scores = []
    sample_iou_scores = []
    sample_f1_scores = []
    sample_precision_scores = []
    sample_recall_scores = []
    sample_accuracy_scores = []
    
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
            
            for i in range(predictions.shape[0]):
                pred_slice = predictions[i].cpu().numpy().flatten()
                label_slice = labels[i].cpu().numpy().flatten()
                
                dice = calculate_dice_coefficient(predictions[i], labels[i].float())
                iou = calculate_iou(predictions[i], labels[i].float())
                
                pred_binary = pred_slice.astype(int)
                label_binary = label_slice.astype(int)
                
                accuracy = accuracy_score(label_binary, pred_binary)
                f1 = f1_score(label_binary, pred_binary, average='binary', zero_division=0)
                precision = precision_score(label_binary, pred_binary, average='binary', zero_division=0)
                recall = recall_score(label_binary, pred_binary, average='binary', zero_division=0)
                
                sample_dice_scores.append(dice)
                sample_iou_scores.append(iou)
                sample_f1_scores.append(f1)
                sample_precision_scores.append(precision)
                sample_recall_scores.append(recall)
                sample_accuracy_scores.append(accuracy)
                
                all_predictions.extend(pred_binary)
                all_targets.extend(label_binary)
    
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
    
    all_predictions = np.array(all_predictions).astype(int)
    all_targets = np.array(all_targets).astype(int)
    confmat = confusion_matrix(all_targets, all_predictions)
    mean_metrics['confusion_matrix'] = confmat
    
    return mean_metrics

def print_test_results(test_results):
    """Print sample-wise test results"""
    print("\n" + "=" * 60)
    print("    SAMPLE-WISE MODEL EVALUATION RESULTS (2.5D)")
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
        
        # For visualization, use middle channel (current slice)
        img_cpu = img_tensor[1].cpu().numpy()  # Middle slice (current)
        label_cpu = label_tensor.cpu().numpy()
        pred_cpu = (torch.sigmoid(output_tensor) > optimal_threshold).squeeze().cpu().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_cpu, cmap='bone')
        axes[0].set_title(f'MRI Slice (Sample {sample_idx}) - 2.5D')
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
    print("STARTING DATA LOADING (2.5D)")
    print("="*60)
    
    try:
        positive_cases, negative_cases = load_full_dataset(
            config.labels_root,
            config.mri_root,
            num_positive=config.num_positive_to_use,
            num_negative=config.num_negative_to_use
        )
        print(f"✓ Loaded {len(positive_cases)} positive and {len(negative_cases)} negative cases")
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return
    
    try:
        splits = get_stratified_data_splits(
            positive_cases, 
            negative_cases,
            train_split=0.70,
            val_split=0.15,
            test_split=0.15
        )
        print(f"✓ Created stratified splits")
    except Exception as e:
        print(f"ERROR creating splits: {e}")
        return
    
    augmentation = MedicalImageAugmentation(
        rotation_angle=15,
        enable_elastic=False,
        enable_intensity=True,
        intensity_variation=0.1
    )
    print(f"✓ Created augmentation pipeline")
    
    try:
        # 🆕 Use 2.5D dataset class instead of 2D
        train_dataset = T2WDataset2p5D(splits['train'], transform=augmentation, slice_axis=0, context_slices=config.context_slices)
        val_dataset = T2WDataset2p5D(splits['val'], transform=None, slice_axis=0, context_slices=config.context_slices)
        test_dataset = T2WDataset2p5D(splits['test'], transform=None, slice_axis=0, context_slices=config.context_slices)
        print(f"✓ Created 2.5D datasets: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    except Exception as e:
        print(f"ERROR creating datasets: {e}")
        return
    
    try:
        dataloaders = get_dataloaders(
            train_dataset, 
            val_dataset, 
            test_dataset,
            use_weighted_sampling=True,
            positive_weight=3.0,
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        test_loader = dataloaders['test']
        print(f"✓ Created data loaders with weighted sampling")
    except Exception as e:
        print(f"ERROR creating data loaders: {e}")
        return
    
    print("\n" + "="*60)
    print("STARTING TRAINING (2.5D)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 🆕 Use 2.5D model with 3-channel input
    model = UNet2D_2p5D(in_ch=3, out_ch=1).to(device)
    
    criterion = BoundaryDiceFocalLoss(
        alpha=config.alpha,
        gamma=config.gamma,
        boundary_weight=config.boundary_weight,
        dice_weight=config.dice_weight,
        focal_weight=config.focal_weight,
        pos_weight=config.pos_weight 
    )
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_loss = float('inf')
    patience_counter = 0
    checkpoint_path = os.path.join(config.checkpoint_dir, "best_model_2p5d.pth")
    
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
    
    if os.path.exists(checkpoint_path):
        print(f"Loading best model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    optimal_threshold = find_optimal_threshold(model, val_loader, device)
    
    print(f"\n{'='*60}")
    print("TESTING WITH OPTIMAL THRESHOLD")
    print(f"{'='*60}")
    test_results = test_model_comprehensive(model, test_loader, device, threshold=optimal_threshold)
    print_test_results(test_results)
    
    print(f"\n{'='*60}")
    print("TESTING WITH STANDARD THRESHOLD 0.5")
    print(f"{'='*60}")
    test_results_standard = test_model_comprehensive(model, test_loader, device, threshold=0.5)
    print_test_results(test_results_standard)
    
    results_path = os.path.join(config.checkpoint_dir, "test_results_2p5d.json")
    results_to_save = {
        'model_version': '2.5D',
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
    
    visualize_predictions(model, test_dataset, device, optimal_threshold, num_samples=3)
    
    print(f"\n{'='*60}")
    print("ALL DONE (2.5D)")
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
