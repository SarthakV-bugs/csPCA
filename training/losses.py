import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryDiceFocalLoss(nn.Module):
    """
    Combined loss for binary segmentation with class imbalance handling.
    
    Components:
    1. Dice Loss: Good for segmentation, handles class imbalance
    2. Focal Loss: Reduces loss for well-classified examples, focuses on hard negatives
    3. Boundary Loss: Encourages accurate boundary predictions
    4. Class weighting: Weights positive class more heavily due to imbalance
    
    For prostate cancer: ~28% positive pixels, ~72% negative pixels
    """
    
    def __init__(self, alpha=0.75, gamma=2.0, boundary_weight=0.2, 
                 dice_weight=0.4, focal_weight=0.4, pos_weight=2.5):
        """
        Args:
            alpha: Focal loss weighting parameter (default: 0.75)
            gamma: Focal loss focusing parameter (default: 2.0)
            boundary_weight: Weight for boundary loss component
            dice_weight: Weight for dice loss component
            focal_weight: Weight for focal loss component
            pos_weight: Weight for positive class in BCE (imbalance ratio)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.boundary_weight = boundary_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.pos_weight = pos_weight

    def dice_loss(self, preds, targets, smooth=1e-6):
        """Dice coefficient loss for segmentation."""
        intersection = (preds * targets).sum()
        dice = (2 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
        return 1 - dice

    def focal_loss(self, preds, targets):
        """
        Focal loss: Reduces loss from well-classified examples (high confidence)
        and focuses on hard examples (low confidence).
        Helps with class imbalance by down-weighting easy negatives.
        """
        # Standard binary cross entropy
        bce = F.binary_cross_entropy(preds, targets, reduction='none')
        
        # Focal loss: weight by (1-p_t)^gamma where p_t is model confidence
        # For positive targets (1): p_t = preds
        # For negative targets (0): p_t = 1 - preds
        p_t = torch.where(targets == 1, preds, 1 - preds)
        
        # Apply focal weighting and alpha balancing
        focal = self.alpha * (1 - p_t) ** self.gamma * bce
        return focal.mean()

    def boundary_loss(self, preds, targets):
        """
        Emphasizes predictions at the boundary between positive and negative regions.
        Uses gradient magnitude to identify boundaries.
        """
        # Compute spatial gradients
        dy = torch.abs(targets[:, :, 1:, :] - targets[:, :, :-1, :])
        dx = torch.abs(targets[:, :, :, 1:] - targets[:, :, :, :-1])
        
        # Create boundary map matching original size by padding
        boundary_y = F.pad(dy, (0, 0, 1, 0))  # Pad bottom
        boundary_x = F.pad(dx, (1, 0, 0, 0))  # Pad right
        
        # Combine to get boundary map (1 if edge, 0 otherwise)
        boundary = ((boundary_y + boundary_x) > 0).float()
        
        # Penalize prediction errors at boundaries more heavily
        boundary_bce = F.binary_cross_entropy(preds, targets, reduction='none')
        boundary_weighted = boundary_bce * (1 + 2 * boundary)
        
        return boundary_weighted.mean()

    def forward(self, preds, targets):
        """
        Combined loss computation.
        
        Args:
            preds: Model predictions (raw logits or sigmoid 0-1), shape: (B, 1, H, W)
            targets: Ground truth binary labels (0 or 1), shape: (B, 1, H, W)
        
        Returns:
            Combined loss value
        """
        # Apply sigmoid if not already done
        preds = torch.sigmoid(preds)
        
        # Ensure targets are float
        targets = targets.float()
        
        # Compute individual loss components
        dice = self.dice_loss(preds, targets)
        focal = self.focal_loss(preds, targets)
        boundary = self.boundary_loss(preds, targets)
        
        # Apply pos_weight to help with class imbalance
        # Weight positive class errors more heavily
        weighted_focal = focal * self.pos_weight / (1 + self.pos_weight)
        
        # Combine weighted losses
        total_loss = (
            self.dice_weight * dice +
            self.focal_weight * weighted_focal +
            self.boundary_weight * boundary
        )
        
        return total_loss
