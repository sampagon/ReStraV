from pathlib import Path
import random
from tqdm import tqdm
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import dinov2_features as d2

def find_videos(root, keys=None, limit=None, patterns=("*.mp4",)):
    root = Path(root)
    paths = [p for pat in patterns for p in root.rglob(pat)]
    if not keys:
        random.shuffle(paths)
        return paths[:limit] if limit is not None else paths
    results = {}
    for k in keys:
        subset = [p for p in paths if k.lower() in str(p).lower()]
        random.shuffle(subset)
        results[k] = subset[:limit] if limit is not None else subset
    return results

def embed_frames(video_paths, device=None, batch_size=1):
    all_Z = []
    for i in tqdm(range(0, len(video_paths), batch_size), desc="Encoding videos", ncols=80):
        batch = video_paths[i:i+batch_size]
        Z = d2.extract_dinov2_embeddings(batch, device=device)
        all_Z.append(Z.cpu())
    return torch.cat(all_Z, dim=0)