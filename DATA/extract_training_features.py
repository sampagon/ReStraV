import torch
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import sys

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import dinov2_features as d2

batch_size = 128
device = "cuda:1" if torch.cuda.is_available() else "cpu"

real_root = Path("../DATA/TRAINING_DATA/REAL")
fake_root = Path("../DATA/TRAINING_DATA/FAKE")
output_h5 = Path("training_features.h5")

real_videos = sorted(real_root.rglob("*.mp4"))
fake_videos = sorted(fake_root.rglob("*.mp4"))
all_videos = [(str(p), 1) for p in real_videos] + [(str(p), 0) for p in fake_videos]

print(f"Found {len(real_videos)} real and {len(fake_videos)} fake videos.")

with h5py.File(output_h5, "w") as h5f:
    dt = h5py.special_dtype(vlen=str)
    path_ds = h5f.create_dataset("path", (len(all_videos),), dtype=dt)
    label_ds = h5f.create_dataset("label", (len(all_videos),), dtype="i")
    feat_ds = h5f.create_dataset("features", (len(all_videos), 21), dtype="f")

    for idx in tqdm(range(0, len(all_videos), batch_size), desc="Extracting features"):
        batch_items = all_videos[idx:idx+batch_size]
        batch_paths = [p for p, _ in batch_items]
        batch_labels = [l for _, l in batch_items]

        Z = d2.extract_dinov2_embeddings(batch_paths, device=device)
        feats = d2.features_from_Z(Z).cpu().numpy()

        for j, (path, label, f) in enumerate(zip(batch_paths, batch_labels, feats)):
            pos = idx + j
            path_ds[pos] = path
            label_ds[pos] = label
            feat_ds[pos, :] = f

print(f"Saved features for {len(all_videos)} videos to {output_h5}")
