import os
import torch
import faiss
import numpy as np
from config import IMG_DIR, CAPTION_FILE


# ─────────────────────────────────────────
# ==== 1. Load image features ====
image_feature_path = "outputs/image_features.pt"
image_path_txt = "outputs/image_paths.txt"
image_features = torch.load(image_feature_path)  

# Convert to float32 for FAISS
image_features_np = image_features.numpy().astype("float32")

with open(image_path_txt, 'r', encoding='utf-8') as f:
    image_paths = [line.strip() for line in f]


dimension = image_features_np.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity if normalized)
index.add(image_features_np)
print(f"✅ Indexed {index.ntotal} image vectors")

# ─────────────────────────────────────────
# ==== 4. Save index ====
os.makedirs("outputs", exist_ok=True)
index_path = "outputs/image.index"
faiss.write_index(index, index_path)
print(f"💾 Saved FAISS index to {index_path}")

# ─────────────────────────────────────────
# ==== (Optional) Save image → index mapping ====
mapping_path = "outputs/image_id_map.txt"
with open(mapping_path, 'w', encoding='utf-8') as f:
    for i, path in enumerate(image_paths):
        f.write(f"{i}\t{path}\n")
