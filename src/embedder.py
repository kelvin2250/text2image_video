import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from transformers import CLIPTokenizer, CLIPModel
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from config import DATA_DIR, CAPTION_FILE

# ────────────────────────────────────────
# Load model + tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model.eval()
print("✅ Loaded CLIP model on", device)

# Preprocess images (ViT-B/32 standard)
preprocess = Compose([
    Resize(224, interpolation=Image.BICUBIC),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
              std=[0.26862954, 0.26130258, 0.27577711])
])

from config import DATA_DIR, CAPTION_FILE  # 👈 dòng này hiện chưa dùng CAPTION_FILE

# 👇 THÊM dòng này ngay sau load config
caption_file = DATA_DIR / "captions.txt"   # hoặc dùng sẵn CAPTION_FILE nếu config đã có

# ────────────────────────────────────────
def load_caption_groups(caption_file):
    image_to_captions = defaultdict(list)
    with open(caption_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rel_path, caption = line.strip().split('\t')
                abs_path = str(DATA_DIR / rel_path)  # 👈 sử dụng đúng base Kaggle path
                image_to_captions[abs_path].append(caption)
            except ValueError:
                print(f"❌ Skipping line: {line.strip()}")
    return dict(image_to_captions)

def truncate_caption(caption, tokenizer, max_len=77):
    tokens = tokenizer.tokenize(caption)
    if len(tokens) > max_len - 2:
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

def extract_features(image_to_captions, device, tokenizer, model, preprocess, batch_size=64):
    all_image_paths = list(image_to_captions.keys())
    image_features, text_features, text_to_image_map = [], [], []

    for i in tqdm(range(0, len(all_image_paths), batch_size), desc="Extracting features"):
        batch_paths = all_image_paths[i:i + batch_size]

        # Encode image
        try:
            imgs = [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
            imgs = torch.stack(imgs).to(device)
            with torch.no_grad():
                img_feats = model.get_image_features(pixel_values=imgs)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            image_features.append(img_feats.cpu())
        except Exception as e:
            print(f"⚠️ Error encoding images {i}-{i+batch_size}: {e}")
            continue

        # Gather and encode captions
        captions, owners = [], []
        for path in batch_paths:
            for cap in image_to_captions.get(path, []):
                captions.append(cap)
                owners.append(path)

        try:
            cap_feats = encode_caption_batch(captions, device, tokenizer, model, batch_size=batch_size)
            text_features.append(cap_feats)
            text_to_image_map.extend(owners)
        except Exception as e:
            print(f"⚠️ Error encoding captions for batch {i}-{i+batch_size}: {e}")

    return (
        torch.cat(image_features, dim=0),
        torch.cat(text_features, dim=0),
        all_image_paths,
        text_to_image_map
    )

def save_features(image_feats, text_feats, image_paths, text_to_image_map, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(image_feats, os.path.join(output_dir, "image_features.pt"))
    torch.save(text_feats, os.path.join(output_dir, "text_features.pt"))

    with open(os.path.join(output_dir, "image_paths.txt"), 'w', encoding='utf-8') as f:
        for p in image_paths:
            f.write(p + '\n')

    with open(os.path.join(output_dir, "caption_map.json"), 'w', encoding='utf-8') as f:
        json.dump(text_to_image_map, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved features to {output_dir}/")

# ────────────────────────────────────────
if __name__ == "__main__":
    image_to_captions = load_caption_groups(caption_file)
    print(f"🔹 Total unique images: {len(image_to_captions)}")

    image_feats, text_feats, image_paths, text_to_image_map = extract_features(
        image_to_captions=image_to_captions,
        device=device,
        tokenizer=tokenizer,
        model=model,
        preprocess=preprocess,
        batch_size=64
    )

    save_features(image_feats, text_feats, image_paths, text_to_image_map)
