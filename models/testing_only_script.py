
# ========================================================================
# TESTING ONLY VERSION - Use this if you already have a trained model
# ========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add your existing UNet2D class definition here (copy from your original code)
# ... (UNet2D class definition) ...

# Add your existing dataset and dataloader setup here
# ... (T2WDataset2D, dataloaders) ...

def calculate_dice_coefficient(pred, target, smooth=1e-8):
    """Calculate Dice coefficient for binary segmentation"""
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()

def calculate_iou(pred, target, smooth=1e-8):
    """Calculate Intersection over Union (IoU)"""
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def test_model_comprehensive(model, test_loader, device, threshold=0.5):
    """
    Test your already trained model - NO RETRAINING NEEDED!
    """
    model.eval()  # Set to evaluation mode

    all_predictions = []
    all_targets = []
    dice_scores = []
    iou_scores = []

    print(f"Testing model with threshold={threshold}...")
    print("🔄 Processing test set... (This may take a few minutes)")

    with torch.no_grad():  # No gradient computation needed
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)

            # Get model predictions
            outputs = model(images)
            predictions = (outputs > threshold).float().squeeze(1)

            # Store for global metrics
            pred_flat = predictions.cpu().numpy().flatten().astype(int)
            label_flat = labels.cpu().numpy().flatten().astype(int)
            all_predictions.extend(pred_flat)
            all_targets.extend(label_flat)

            # Calculate per-image metrics
            batch_size = predictions.shape[0]
            for i in range(batch_size):
                pred_slice = predictions[i]
                label_slice = labels[i].float()

                dice = calculate_dice_coefficient(pred_slice, label_slice)
                iou = calculate_iou(pred_slice, label_slice)
                dice_scores.append(dice)
                iou_scores.append(iou)

    # Calculate global metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    accuracy = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions, average='binary', zero_division=0)
    precision = precision_score(all_targets, all_predictions, average='binary', zero_division=0)
    recall = recall_score(all_targets, all_predictions, average='binary', zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_targets)

    # Calculate averages
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'dice_coefficient': avg_dice,
        'iou_score': avg_iou,
        'confusion_matrix': cm,
        'dice_scores': dice_scores,
        'iou_scores': iou_scores
    }

def print_test_results(results):
    """Print formatted test results"""
    print("\n" + "="*50)
    print("         MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy        : {results['accuracy']:.4f} ({results['accuracy']*100:.1f}%)")
    print(f"F1 Score        : {results['f1_score']:.4f}")
    print(f"Precision       : {results['precision']:.4f}")
    print(f"Recall          : {results['recall']:.4f}")
    print(f"Dice Coefficient: {results['dice_coefficient']:.4f}")
    print(f"IoU Score       : {results['iou_score']:.4f}")

    cm = results['confusion_matrix']
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"\nConfusion Matrix:")
        print(f"True Positives : {tp:,}")
        print(f"True Negatives : {tn:,}")
        print(f"False Positives: {fp:,}")
        print(f"False Negatives: {fn:,}")

        # Additional derived metrics
        if (tp + fp) > 0 and (tn + fp) > 0:
            specificity = tn / (tn + fp)
            print(f"\nDerived Metrics:")
            print(f"Sensitivity (Recall): {results['recall']:.4f}")
            print(f"Specificity        : {specificity:.4f}")

    print("="*50)

# ============================================================================
# USAGE: Replace the end of your training script with this:
# ============================================================================

# AFTER your training is complete, add this:
print("\n🎯 TESTING YOUR TRAINED MODEL...")
print("(No retraining needed - using existing model weights)")

# Run comprehensive evaluation
test_results = test_model_comprehensive(model, test_loader, device, threshold=0.5)

# Print results
print_test_results(test_results)

# Quick visualization
def quick_visualization(model, test_dataset, device, num_samples=3):
    """Quick visualization of results"""
    model.eval()

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Get random sample
        idx = np.random.randint(0, len(test_dataset))
        img_tensor, label_tensor = test_dataset[idx]

        with torch.no_grad():
            input_tensor = img_tensor.unsqueeze(0).to(device)
            output = model(input_tensor)
            pred = (output > 0.5).squeeze().cpu().numpy()

        img = img_tensor.squeeze().cpu().numpy()
        label = label_tensor.cpu().numpy()

        # Calculate metrics
        pred_tensor = torch.from_numpy(pred).float()
        label_tensor_calc = torch.from_numpy(label).float()
        dice = calculate_dice_coefficient(pred_tensor, label_tensor_calc)
        iou = calculate_iou(pred_tensor, label_tensor_calc)

        # Plot
        axes[i, 0].imshow(img, cmap='bone')
        axes[i, 0].set_title(f'Input Image {idx}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(label, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title(f'Prediction\nDice: {dice:.3f}')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

# Show quick visualization
quick_visualization(model, test_dataset, device, num_samples=3)

print("\n✅ EVALUATION COMPLETE!")
print(f"📊 Your model achieved:")
print(f"   • Accuracy: {test_results['accuracy']:.1%}")
print(f"   • Dice Score: {test_results['dice_coefficient']:.3f}")
print(f"   • F1 Score: {test_results['f1_score']:.3f}")

# Interpretation guide
dice_score = test_results['dice_coefficient']
if dice_score >= 0.90:
    performance = "🟢 EXCELLENT"
elif dice_score >= 0.80:
    performance = "🟡 GOOD"
elif dice_score >= 0.70:
    performance = "🟠 MODERATE"
else:
    performance = "🔴 NEEDS IMPROVEMENT"

print(f"\n🎯 Performance Assessment: {performance}")
print(f"   Dice coefficient: {dice_score:.3f}")

if dice_score < 0.70:
    print("\n💡 Suggestions for improvement:")
    print("   • Try training for more epochs")
    print("   • Adjust learning rate or optimizer")
    print("   • Consider data augmentation")
    print("   • Check for class imbalance issues")
