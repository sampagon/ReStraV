import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pathlib import Path
import random
from tqdm import tqdm
import sys

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

batch_size = 64
device = "cuda:2"

real_root = Path("DATA/TRAINING_DATA/REAL")
fake_root = Path("DATA/TRAINING_DATA/FAKE")

real = find_videos(real_root, limit=60)
fake = find_videos(fake_root, keys=["pika", "t2vz", "vc2", "ms"], limit=60)
fake_all = [p for lst in fake.values() for p in lst]

print("Extracting REAL frame embeddings...")
Z_real = embed_frames(real, device=device, batch_size=batch_size)
print("Extracting FAKE frame embeddings...")
Z_fake = embed_frames(fake_all, device=device, batch_size=batch_size)

Nr, T, D = Z_real.shape
Nf = Z_fake.shape[0]
N  = Nr + Nf
print(f"Shapes -> real: {Z_real.shape}, fake: {Z_fake.shape}")

d_r, th_r = d2.compute_temporal_geometry(Z_real)
d_f, th_f = d2.compute_temporal_geometry(Z_fake)

X_steps_real = torch.stack([d_r[:, :-1], th_r], dim=-1) 
X_steps_fake = torch.stack([d_f[:, :-1], th_f], dim=-1) 

Tm = T - 2 
X_steps = torch.cat([X_steps_real, X_steps_fake], dim=0)  
X_tsne  = X_steps.reshape(N * Tm, 2).numpy()       

# Labels to color and to connect each video's steps
video_labels = np.array([0]*Nr + [1]*Nf)     
step_labels  = np.repeat(video_labels, Tm)                
video_ids    = np.repeat(np.arange(N), Tm)             

print("Running t-SNE on per-step geometry (d, θ)…")
tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
X_emb = tsne.fit_transform(X_tsne)                  

plt.figure(figsize=(7, 6))
c_real, c_fake = "blue", "red"

for vid in range(N):
    mask = (video_ids == vid)
    pts  = X_emb[mask]
    c    = c_real if vid < Nr else c_fake
    plt.plot(pts[:, 0], pts[:, 1], '-', lw=0.6, alpha=0.5, color=c)
    plt.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.6, color=c)

# legend proxies
plt.scatter([], [], s=20, color=c_real, label="Natural")
plt.scatter([], [], s=20, color=c_fake, label="Synthetic")

plt.legend()
plt.title("t-SNE Trajectories of Per-step Geometry (distance & curvature)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()

out_path = Path("tsne_curvature_trajectories.png")
plt.savefig(out_path, dpi=300)
plt.close()
print(f"Saved t-SNE plot to: {out_path.resolve()}")
