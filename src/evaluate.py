import torch
import torch.nn.functional as F
import json
from pathlib import Path
from config import OUTPUT_DIR

# ────────────────────────────────────────
# Config paths
eval_dir = Path(OUTPUT_DIR)
image_feat_path = eval_dir / "image_features.pt"
text_feat_path  = eval_dir / "text_features.pt"
caption_map_path = eval_dir / "caption_map.json"
image_paths_txt  = eval_dir / "image_paths.txt"

# ────────────────────────────────────────
# 1. Load features and caption mapping
def load_data():
    if not image_feat_path.exists() or not text_feat_path.exists():
        raise FileNotFoundError(f"❌ Feature files missing in {eval_dir}")
    image_feats = torch.load(image_feat_path)
    text_feats  = torch.load(text_feat_path)

    with open(caption_map_path, 'r', encoding='utf-8') as f:
        caption_map = json.load(f)

    image_paths = [line.strip() for line in open(image_paths_txt, 'r', encoding='utf-8')]
    return image_feats, text_feats, caption_map, image_paths

# ────────────────────────────────────────
# 2. Normalize features
def normalize_feats(image_feats, text_feats):
    image_feats = F.normalize(image_feats, p=2, dim=1)
    text_feats  = F.normalize(text_feats, p=2, dim=1)
    return image_feats, text_feats

# ────────────────────────────────────────
# 3. Compute ground-truth indices
def build_gt_indices(caption_map, image_paths):
    path_to_idx = {p: idx for idx, p in enumerate(image_paths)}
    gt_indices = [path_to_idx[path] for path in caption_map]
    return gt_indices

# ────────────────────────────────────────
# 4. Recall@K computation
def compute_recall_at_k(sim_matrix, gt_indices, k=1):
    topk = sim_matrix.topk(k, dim=1).indices
    gt_tensor = torch.tensor(gt_indices).unsqueeze(1)
    correct = torch.any(topk == gt_tensor, dim=1)
    return correct.float().mean().item()

# ────────────────────────────────────────
# 5. Main evaluation
def main():
    image_feats, text_feats, caption_map, image_paths = load_data()
    image_feats, text_feats = normalize_feats(image_feats, text_feats)

    sim = text_feats @ image_feats.T
    gt_inds = build_gt_indices(caption_map, image_paths)

    for k in [1, 5, 10]:
        recall = compute_recall_at_k(sim, gt_inds, k)
        print(f"📈 Recall@{k}: {recall * 100:.2f}%")

if __name__ == "__main__":
    main()
