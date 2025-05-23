import os
import faiss
import torch
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from config import DATA_DIR, OUTPUT_DIR

# ────────────────────────────────────────
# Paths (auto for local/Kaggle)
OUTPUT_DIR = Path(OUTPUT_DIR)
INDEX_PATH = OUTPUT_DIR / "faiss.index"
IMAGE_LIST_PATH = OUTPUT_DIR / "image_paths.txt"

# ────────────────────────────────────────
# 1. Load CLIP model + processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
print(f"✅ Loaded CLIP model on {device}")

# ────────────────────────────────────────
# 2. Load FAISS index & image paths
index = faiss.read_index(str(INDEX_PATH))

with open(IMAGE_LIST_PATH, "r", encoding="utf-8") as f:
    image_paths = [line.strip() for line in f]

print(f"📁 Loaded {len(image_paths)} image paths")

# 3. Encode text query
def encode_query(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_feat = model.get_text_features(**inputs)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    return text_feat.cpu().numpy().astype("float32")

# ────────────────────────────────────────
# 4. Search top-K images
def search_images(query, top_k=5):
    q_vec = encode_query(query)
    D, I = index.search(q_vec, top_k)
    results = []
    for idx in I[0]:
        path = image_paths[idx]
        results.append(path)
    return results

# ────────────────────────────────────────
# 5. Display results
def show_images(paths, query):
    imgs = []
    for p in paths:
        try:
            imgs.append(Image.open(p))
        except Exception as e:
            print(f"❌ Could not open {p}: {e}")
    plt.figure(figsize=(15, 5))
    for i, img in enumerate(imgs):
        plt.subplot(1, len(imgs), i+1)
        plt.imshow(img)
        plt.title(f"Rank {i+1}")
        plt.axis('off')
    plt.suptitle(f"🔍 Query: {query}")
    plt.show()

# ────────────────────────────────────────
# 6. CLI loop
if __name__ == "__main__":
    while True:
        query = input("\n📝 Enter query (or 'exit'): ")
        if query.strip().lower() == 'exit':
            print("👋 Goodbye.")
            break
        try:
            topk = search_images(query, top_k=5)
            print("\n🔍 Top results:")
            for p in topk:
                print(f"📸 {p}")
            show_images(topk, query)
        except Exception as e:
            print(f"❌ Error: {e}")
