from pathlib import Path
import random

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