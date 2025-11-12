import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm
from pathlib import Path

#Load data
h5_path = "../DATA/training_features.h5"

with h5py.File(h5_path, "r") as f:
    X = f["features"][:] 
    y = f["label"][:] 
    paths = f["path"][:].astype(str)

print(f"Loaded {len(X)} samples from {h5_path}")

#Filter fake paths by interested models
allowed_models = [
    "pika_videos_example",
    "vc2_videos_example",
    "t2vz_videos_example",
    "ms_videos_example"
]

mask_real = y == 1
mask_fake = np.array([any(m in p for m in allowed_models) for p in paths]) & (y == 0)

real_stats = X[mask_real][:, -8:]
fake_stats = X[mask_fake][:, -8:]

print(f"Filtered fake samples: {len(fake_stats)} (from allowed models)")
print(f"Real samples: {len(real_stats)}")

#Balance classes (equal priors)
n_balanced = min(len(real_stats), len(fake_stats))
rng = np.random.default_rng(seed=42)
sel_real = rng.choice(len(real_stats), n_balanced, replace=False)
sel_fake = rng.choice(len(fake_stats), n_balanced, replace=False)

real_stats = real_stats[sel_real]
fake_stats = fake_stats[sel_fake]

print(f"Balanced: {n_balanced} real vs {n_balanced} fake samples")

#Plot
labels = [
    r'$\mu_d$', r'$min_d$', r'$max_d$', r'$\sigma^2_d$',
    r'$\mu_\theta$', r'$min_\theta$', r'$max_\theta$', r'$\sigma^2_\theta$'
]

fig, axes = plt.subplots(2, 4, figsize=(12, 4))
axes = axes.ravel()

for i in tqdm(range(8), desc="Plotting distributions"):
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
print("Saved plot as distributions_dino.png")
