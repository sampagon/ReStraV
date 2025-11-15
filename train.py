import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score, accuracy_score, confusion_matrix
from pathlib import Path
from tqdm import tqdm

#Load data
h5_path = Path("DATA/training_features.h5")

with h5py.File(h5_path, "r") as f:
    X = f["features"][:].astype(np.float32)
    y = f["label"][:].astype(np.int64)
    paths = f["path"][:].astype(str)
    
print(f"Loaded {len(X)} samples from {h5_path}")

#Balance classes (equal priors)
idx_real = np.where(y == 1)[0]
idx_fake = np.where(y == 0)[0]

n_real = len(idx_real)
n_fake = len(idx_fake)
n_balanced = min(n_real, n_fake)

#randomly sample equal numbers from each class
rng = np.random.default_rng(seed=42)
sel_real = rng.choice(idx_real, n_balanced, replace=False)
sel_fake = rng.choice(idx_fake, n_balanced, replace=False)

balanced_idx = np.concatenate([sel_real, sel_fake])
assert len(balanced_idx) == len(np.unique(balanced_idx)), "Duplicate indices in balanced dataset!"
rng.shuffle(balanced_idx)

X_bal = X[balanced_idx]
y_bal = y[balanced_idx]
paths_bal = paths[balanced_idx]

print(f"Balanced dataset size: {len(X_bal)} (real={n_balanced}, fake={n_balanced})")

#Normalize
mean = X_bal.mean(axis=0, keepdims=True)
std = X_bal.std(axis=0, keepdims=True) + 1e-8
X_bal = (X_bal - mean) / std

#Stratified 50/50 train/test split
X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
    X_bal,
    y_bal,
    paths_bal,
    test_size=0.5,
    stratify=y_bal,
    random_state=42
)

#Convert to tensors and datasets
train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_ds  = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

#Create DataLoaders
batch_size = 512

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

print(f"Train: {len(train_ds)}  Test: {len(test_ds)}  (each balanced 50/50)")

#Sanity check
xb, yb = next(iter(train_loader))
print("Example batch:", xb.shape, yb.shape)
print("Label distribution (train):", torch.bincount(torch.from_numpy(y_train)))
print("Label distribution (test):", torch.bincount(torch.from_numpy(y_test)))

#Train
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
class MLP(nn.Module):
    def __init__(self, in_dim=21, h1=64, h2=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1)
        )
    def forward(self, x):
        return self.net(x)

model = MLP().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        xb, yb = xb.to(device), yb.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    print(f"Epoch {epoch+1} Loss: {running_loss / len(train_loader.dataset):.4f}")

# Optimize threshold τ* on training set (F1)
model.eval()
train_logits, train_labels = [], []
with torch.no_grad():
    for xb, yb in train_loader:
        xb = xb.to(device)
        out = torch.sigmoid(model(xb)).cpu().numpy().ravel()
        train_logits.append(out)
        train_labels.append(yb.numpy())

train_logits = np.concatenate(train_logits)
train_labels = np.concatenate(train_labels)

thresholds = np.linspace(0.1, 0.9, 81)
best_f1, best_tau = 0.0, 0.5
for t in thresholds:
    preds = (train_logits >= t).astype(int)
    f1 = f1_score(train_labels, preds)
    if f1 > best_f1:
        best_f1, best_tau = f1, t

print(f"Best τ*={best_tau:.3f} with F1={best_f1:.3f} on training set")

#Evaluate on test set with τ*
test_logits, test_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        out = torch.sigmoid(model(xb)).cpu().numpy().ravel()
        test_logits.append(out)
        test_labels.append(yb.numpy())

test_logits = np.concatenate(test_logits)
test_labels = np.concatenate(test_labels)
test_preds = (test_logits >= best_tau).astype(int)

precision, recall, f1, _ = precision_recall_fscore_support(
    test_labels, test_preds, average="binary"
)
auc = roc_auc_score(test_labels, test_logits)

acc = accuracy_score(test_labels, test_preds)
cm = confusion_matrix(test_labels, test_preds)

print("\n=== Test Performance ===")
print(f"Accuracy : {acc:.3f}")
print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}  AUC: {auc:.3f}")
print("Confusion matrix (rows=true [0,1], cols=pred [0,1]):")
print(cm)

torch.save(model.state_dict(), "model.pt")
np.save("mean.npy", mean)
np.save("std.npy", std)
np.save("best_tau.npy", best_tau)