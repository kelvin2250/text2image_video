import torch
import torch.nn.functional as F
import json
from pathlib import Path

# ────────────────────────────────────────
# 1. Đường dẫn file
root = Path(__file__).resolve().parent.parent
outputs_dir = root / "outputs"

image_features = torch.load(outputs_dir / "image_features.pt")  # [M, D]
text_features  = torch.load(outputs_dir / "text_features.pt")   # [N, D]

with open(outputs_dir / "caption_map.json", 'r', encoding='utf-8') as f:
    caption_map = json.load(f)

# Normalize feature vectors
image_features = F.normalize(image_features, p=2, dim=1)
text_features  = F.normalize(text_features, p=2, dim=1)

# ────────────────────────────────────────
# 2. Chuẩn bị chỉ số Ground-truth: caption i → ảnh gốc index
# Vì text_to_image_map chứa path, ta tìm index tương ứng trong image_paths
image_path_list = [line.strip() for line in open(outputs_dir / "image_paths.txt", encoding='utf-8')]
image_path_to_index = {p: idx for idx, p in enumerate(image_path_list)}
gt_indices = [image_path_to_index[path] for path in caption_map]  # N dòng ứng với text_features

# ────────────────────────────────────────
# 3. Tính Recall@K
def compute_recall_at_k(similarity_matrix, gt_indices, k=1):
    topk = similarity_matrix.topk(k, dim=1).indices  # [N, k]
    gt_tensor = torch.tensor(gt_indices).unsqueeze(1).to(topk.device)  # [N, 1]
    correct = torch.any(topk == gt_tensor, dim=1)
    return correct.float().mean().item()

# ────────────────────────────────────────
# 4. Tính Similarity matrix (cosine)
similarity = text_features @ image_features.T  # [N_text, N_image]

# ────────────────────────────────────────
# 5. In kết quả Recall@K
for k in [1, 5, 10]:
    recall = compute_recall_at_k(similarity, gt_indices, k=k)
    print(f"📈 Recall@{k}: {recall * 100:.2f}%")
