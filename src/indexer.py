import os
import torch
import faiss
import numpy as np
from pathlib import Path
from config import OUTPUT_DIR

# ─────────────────────────────────────────
# Paths (adapt automatically for local/Kaggle)
OUTPUT_DIR = Path(OUTPUT_DIR)
FEATURE_PATH = OUTPUT_DIR / "image_features.pt"
IMAGE_LIST_PATH = OUTPUT_DIR / "image_paths.txt"
INDEX_PATH = OUTPUT_DIR / "image.index"
MAPPING_PATH = OUTPUT_DIR / "image_id_map.txt"


# ==== 1. Load image features ====
if not FEATURE_PATH.exists():
    raise FileNotFoundError(f"❌ Feature file not found: {FEATURE_PATH}")
image_features = torch.load(FEATURE_PATH)

# Convert to numpy float32 for FAISS
image_features_np = image_features.numpy().astype("float32")

with open(IMAGE_LIST_PATH, 'r', encoding='utf-8') as f:
    image_paths = [line.strip() for line in f]

# ─────────────────────────────────────────
# ==== 2. Build FAISS index ====
dim = image_features_np.shape[1]
index = faiss.IndexFlatIP(dim)  # IP for cosine if features are normalized
index.add(image_features_np)
print(f"✅ Indexed {index.ntotal} image vectors")

# ─────────────────────────────────────────
# ==== 3. Save index and mapping ====
faiss.write_index(index, str(INDEX_PATH))
print(f"💾 Saved FAISS index to {INDEX_PATH}")

# Save mapping from internal ID to image path
with open(MAPPING_PATH, 'w', encoding='utf-8') as f:
    for idx, path in enumerate(image_paths):
        f.write(f"{idx}\t{path}\n")
print(f"💾 Saved image ID mapping to {MAPPING_PATH}")
