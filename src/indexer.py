import os
import torch
import faiss
import numpy as np
from pathlib import Path
from config import OUTPUT_DIR

# ─────────────────────────────────────────
# ==== Constants ====
OUTPUT_DIR = Path(OUTPUT_DIR)
FEATURE_PATH = OUTPUT_DIR / "image_features.pt"
IMAGE_LIST_PATH = OUTPUT_DIR / "image_paths.txt"
INDEX_PATH = OUTPUT_DIR / "faiss.index"
MAPPING_PATH = OUTPUT_DIR / "faiss_id_map.txt"

# ─────────────────────────────────────────
def build_faiss_index(image_features_np):
    dim = image_features_np.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity (after normalization)
    index.add(image_features_np)
    print(f"✅ Indexed {index.ntotal} image vectors with dim = {dim}")
    return index

def save_faiss_mapping(image_paths, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for idx, path in enumerate(image_paths):
            f.write(f"{idx}\t{Path(path).as_posix()}\n")
    print(f"💾 Saved FAISS ID → image path map to {save_path}")

# ─────────────────────────────────────────
if __name__ == "__main__":
    # ==== 1. Load features ====
    assert FEATURE_PATH.exists(), f"❌ Not found: {FEATURE_PATH}"
    assert IMAGE_LIST_PATH.exists(), f"❌ Not found: {IMAGE_LIST_PATH}"

    image_features = torch.load(FEATURE_PATH)
    image_features_np = image_features.numpy().astype("float32")

    with open(IMAGE_LIST_PATH, 'r', encoding='utf-8') as f:
        image_paths = [line.strip() for line in f]

    assert image_features_np.shape[0] == len(image_paths), "❌ Feature count mismatch"

    # ==== 2. Build index ====
    index = build_faiss_index(image_features_np)

    # ==== 3. Save index and mapping ====
    faiss.write_index(index, str(INDEX_PATH))
    print(f"💾 Saved FAISS index to {INDEX_PATH}")

    save_faiss_mapping(image_paths, MAPPING_PATH)
