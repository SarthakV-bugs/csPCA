from pathlib import Path


class Config:
# Paths (update to your local system)
project_root = Path.cwd()
checkpoint_dir = project_root / 'checkpoints'
labels_root = project_root / 'raw_csPCa_data' / 'picai_labels'
mri_root = project_root / 'raw_csPCa_data' / 'mri_images'


# Dataset
num_positive_to_use = 20
num_negative_to_use = 20
target_size = (256, 256)
batch_size = 4
num_workers = 0


# Training
num_epochs = 10
learning_rate = 1e-4
patience = 3


# Loss weights
alpha = 0.75
gamma = 3
boundary_weight = 0.5
dice_weight = 0.25
focal_weight = 0.25


# Sampling
positive_weight = 10.0


# Postprocessing
min_size = 50
max_holes = 30


config = Config()
