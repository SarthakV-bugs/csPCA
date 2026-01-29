from pathlib import Path


class Config:
    # Paths - relative to this config.py file location
    project_root = Path(__file__).parent
    checkpoint_dir = project_root / 'checkpoints'
    labels_root = project_root / 'data' / 'picai_labels'
    mri_root = project_root / 'data' / 'mri_images'

    # Dataset - NOW USING FULL AVAILABLE DATA!
    # Available: ~425 positive cases, ~1075 negative cases (1500 total)
    # Strategy: Use 70% train, 15% val, 15% test with stratified splits
    num_positive_to_use = 300  # ~70% of 425 positive cases
    num_negative_to_use = 750  # ~70% of 1075 negative cases
    target_size = (256, 256)
    batch_size = 8  # Can increase with more data
    num_workers = 4
    
    # Data splits (applied after selecting above cases)
    train_split = 0.70  # 70% for training
    val_split = 0.15   # 15% for validation
    test_split = 0.15  # 15% for testing

    # Training
    num_epochs = 20  # Increased to leverage more data
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
    # pos_weight = number_negative / number_positive
    # Approximate: 1075 / 425 ≈ 2.5
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


config = Config()
