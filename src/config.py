from pathlib import Path
import os

# Detect running on Kaggle
ON_KAGGLE = Path("/kaggle/input").exists()

# Base data directory
if ON_KAGGLE:
    DATASET_DIR = Path("/kaggle/input/flickr30k-images/flickr30k_images")
    CAPTION_FILE = Path("/kaggle/input/flickr30k-images/captions.txt")
    OUTPUT_DIR = Path("/kaggle/working/outputs")
else:
    DATASET_DIR = Path("data/flickr30k_images")
    CAPTION_FILE = Path("data/captions.txt")
    OUTPUT_DIR = Path("outputs")

# Exported to other scripts
DATA_DIR = DATASET_DIR
