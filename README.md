# csPCA 2D Segmentation


Modularized PyTorch project containing:
- UNet2D model
- Boundary + Dice + Focal combined loss
- Dataset utilities for 2D slice extraction
- Training engine with checkpointing & early stopping


Run: `python3 main.py`


Update paths in `config.py` to point to your local MRI and label directories
