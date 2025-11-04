import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from tqdm import tqdm
from scipy.stats import gaussian_kde

from utils import find_videos

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import dinov2_features as d2

batch_size = 64
device = "cuda:2"

real_root = Path("../DATA/TRAINING_DATA/REAL")
fake_root = Path("../DATA/TRAINING_DATA/FAKE")

real = find_videos(real_root, limit=500)
fake = find_videos(fake_root, keys=["pika", "t2vz", "vc2", "ms"], limit=125)
fake_all = [p for lst in fake.values() for p in lst]

print("Extracting REAL frame embeddings...")
Z_real = []
for i in tqdm(range(0, len(real), batch_size), desc="Encoding videos", ncols=80):
    batch = real[i:i+batch_size]
    Z = d2.extract_dinov2_embeddings(batch, device=device)
    Z_real.append(Z.cpu())
Z_real = torch.cat(Z_real, dim=0)

print("Extracting FAKE frame embeddings...")
Z_fake = []
for i in tqdm(range(0, len(real), batch_size), desc="Encoding videos", ncols=80):
    batch = fake_all[i:i+batch_size]
    Z = d2.extract_dinov2_embeddings(batch, device=device)
    Z_fake.append(Z.cpu())
Z_fake = torch.cat(Z_fake, dim=0)

print(f"real shape: {Z_real.shape}")
print(f"fake shape: {Z_fake.shape}")

fake_stats = d2.features_from_Z(Z_fake)[:, -8:]
real_stats = d2.features_from_Z(Z_real)[:, -8:]

labels = [r'$\mu_d$', r'$min_d$', r'$max_d$', r'$\sigma^2_d$',
          r'$\mu_\theta$', r'$min_\theta$', r'$max_\theta$', r'$\sigma^2_\theta$']

fig, axes = plt.subplots(2, 4, figsize=(12, 4))
axes = axes.ravel()

for i in range(8):
    ax = axes[i]
    x_real = real_stats[:, i]
    x_fake = fake_stats[:, i]

    kde_real = gaussian_kde(x_real)
    kde_fake = gaussian_kde(x_fake)

    xmin = min(x_real.min(), x_fake.min())
    xmax = max(x_real.max(), x_fake.max())
    xs = np.linspace(xmin, xmax, 300)

    ax.fill_between(xs, kde_real(xs), alpha=0.4, color='blue', label='Natural')
    ax.fill_between(xs, kde_fake(xs), alpha=0.4, color='red', label='AI-Generated')

    ax.plot(xs, kde_real(xs), color='blue', lw=1.5)
    ax.plot(xs, kde_fake(xs), color='red', lw=1.5)

    ax.set_title(labels[i], fontsize=12)
    if i % 4 == 0:
        ax.set_ylabel("Density")
    ax.set_xlabel("Value")

fig.legend(['Natural', 'AI-Generated'], loc='lower center', ncol=2, fontsize=12)

plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.savefig("distributions_dino.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved plot as .png")
