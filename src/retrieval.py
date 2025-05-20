import os
import faiss
import torch
import json
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from config import IMG_DIR, CAPTION_FILE

# ────────────────────────────────────────
# 1. Load CLIP model + processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
print(f"✅ Loaded CLIP model on {device}")

# ────────────────────────────────────────
# 2. Load FAISS index & image paths
index_path = "outputs/image.index"
path_file = "outputs/image_paths.txt"

index = faiss.read_index(index_path)

with open(path_file, "r", encoding="utf-8") as f:
    image_paths = [line.strip() for line in f]

print(f"📁 Loaded {len(image_paths)} image paths")

# ────────────────────────────────────────
# 3. Encode text query
def encode_query(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_feat = model.get_text_features(**inputs)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    return text_feat.cpu().numpy().astype("float32")

# ────────────────────────────────────────
# 4. Search top-K image
def search_image(query, top_k=5):
    query_vector = encode_query(query)
    D, I = index.search(query_vector, top_k)
    results = []
    for idx in I[0]:
        path = image_paths[idx]
        if not os.path.exists(path):
            print(f"⚠️ Warning: File not found: {path}")
            continue
        results.append(path)
    return results

# ────────────────────────────────────────
# 5. Show results
def show_images(image_paths, query):
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"❌ Could not open image {path}: {e}")
            continue

    plt.figure(figsize=(15, 5))
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img)
        plt.title(f"Top {i+1}")
        plt.axis("off")

    plt.suptitle(f"🔍 Query: {query}", fontsize=14)
    plt.tight_layout()
    plt.show()

# ────────────────────────────────────────
# 6. CLI interface
if __name__ == "__main__":
    while True:
        try:
            query = input("\n📝 Enter your text query (or 'exit'): ")
            if query.strip().lower() == "exit":
                print("👋 Exiting.")
                break

            top_paths = search_image(query, top_k=5)

            print("\n🔍 Top retrieved images:")
            for path in top_paths:
                print(f"📸 {path}")

            show_images(top_paths, query)

        except KeyboardInterrupt:
            print("\n👋 Exiting by KeyboardInterrupt.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
