import os
import json
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from transformers import CLIPTokenizer, CLIPModel
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from config import DATA_DIR, CAPTION_FILE, OUTPUT_DIR

# ────────────────────────────────────────────────────────────────
# Load CLIP model and tokenizer
# ────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model.eval()
print(f"✅ CLIP model loaded on {device}")

# Image preprocessing pipeline (ViT-B/32 standard)
preprocess = Compose([
    Resize(224, interpolation=Image.BICUBIC),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
              std=[0.26862954, 0.26130258, 0.27577711])
])

# ────────────────────────────────────────────────────────────────
# Load captions from file, group them by image path
# ────────────────────────────────────────────────────────────────
def load_caption_groups(caption_file):
    image_to_captions = defaultdict(list)
    with open(caption_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rel_path, caption = line.strip().split('\t')
                rel_path = rel_path.replace("\\", "/")
                abs_path = str(DATA_DIR / Path(rel_path).name)
                image_to_captions[abs_path].append(caption)
            except ValueError:
                print(f"❌ Skipping malformed line: {line.strip()}")
    return dict(image_to_captions)

# ────────────────────────────────────────────────────────────────
# Truncate long captions and encode them to embeddings
# ────────────────────────────────────────────────────────────────
def truncate_caption(caption, tokenizer, max_len=77):
    tokens = tokenizer.tokenize(caption)
    tokens = tokens[:max_len - 2]
    return tokenizer.convert_tokens_to_string(tokens)

def encode_caption_batch(captions, device, tokenizer, model, batch_size=128):
    all_feats = []
    processed = [truncate_caption(c, tokenizer) for c in captions]

    for i in range(0, len(processed), batch_size):
        batch = processed[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        all_feats.append(outputs.cpu())

    return torch.cat(all_feats, dim=0)

# ────────────────────────────────────────────────────────────────
# Extract image and caption embeddings in batch
# ────────────────────────────────────────────────────────────────
def extract_features(image_to_captions, device, tokenizer, model, preprocess, batch_size=64):
    image_features = []
    text_features = []
    image_path_list = []
    caption_map = []

    all_image_paths = list(image_to_captions.keys())
    for i in tqdm(range(0, len(all_image_paths), batch_size), desc="Extracting features"):
        batch_paths = all_image_paths[i:i + batch_size]

        try:
            imgs = [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
            imgs = torch.stack(imgs).to(device)
            with torch.no_grad():
                img_feats = model.get_image_features(pixel_values=imgs)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            image_features.append(img_feats.cpu())
            image_path_list.extend([Path(p).as_posix() for p in batch_paths])
        except Exception as e:
            print(f"⚠️ Failed image batch {i}-{i+batch_size}: {e}")
            continue

        # Process captions for this image batch
        captions = []
        for path in batch_paths:
            for cap in image_to_captions.get(path, []):
                captions.append(cap)
                caption_map.append({"caption": cap, "image_path": Path(path).as_posix()})

        try:
            cap_feats = encode_caption_batch(captions, device, tokenizer, model, batch_size=batch_size)
            text_features.append(cap_feats)
        except Exception as e:
            print(f"⚠️ Failed caption batch {i}-{i+batch_size}: {e}")

    return {
        "image_features": torch.cat(image_features, dim=0),  # Tensor [num_images, D]
        "text_features": torch.cat(text_features, dim=0),    # Tensor [num_captions, D]
        "image_paths": image_path_list,                      # List[str]
        "caption_map": caption_map                           # List[Dict]
    }

# ────────────────────────────────────────────────────────────────
# Save all output features to disk
# ────────────────────────────────────────────────────────────────
def save_features(output, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(output["image_features"], output_dir / "image_features.pt")
    torch.save(output["text_features"], output_dir / "text_features.pt")

    with open(output_dir / "image_paths.txt", "w", encoding="utf-8") as f:
        for path in output["image_paths"]:
            f.write(path + "\n")

    with open(output_dir / "caption_map.json", "w", encoding="utf-8") as f:
        json.dump(output["caption_map"], f, indent=2, ensure_ascii=False)

    print(f"✅ Features saved to: {output_dir}/")

# ────────────────────────────────────────────────────────────────
# Main pipeline
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    image_to_captions = load_caption_groups(CAPTION_FILE)
    print(f"🔹 Total unique images: {len(image_to_captions)}")

    output = extract_features(
        image_to_captions=image_to_captions,
        device=device,
        tokenizer=tokenizer,
        model=model,
        preprocess=preprocess,
        batch_size=64
    )

    save_features(output, OUTPUT_DIR)
