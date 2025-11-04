import os
import tarfile
import glob
import tempfile
import concurrent.futures
from pathlib import Path
from urllib.request import urlopen

import requests
from tqdm import tqdm
from huggingface_hub import snapshot_download

TRAINING_ROOT = Path("TRAINING_DATA")
REAL_DIR = TRAINING_ROOT / "REAL"
FAKE_DIR = TRAINING_ROOT / "FAKE"

LIST_URL = "https://dl.fbaipublicfiles.com/video_similarity_challenge/46ef53734a4/vsc_url_list.txt"
REF_LIST = "ref_file_paths.txt"
MAX_WORKERS = 32
TIMEOUT = 30

VIDPROM_REPO = "WenhaoWang/VidProM"

def download_file(url, dest_root):
    filename = Path(url).name
    dest_root.mkdir(parents=True, exist_ok=True)
    dest_path = dest_root / filename

    if dest_path.exists():
        return False

    try:
        resp = requests.get(url, stream=True, timeout=TIMEOUT)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        tqdm.write(f"Failed: {filename} ({e})")
        return False

def download_real():
    print("\n=== DOWNLOADING REAL VIDEOS ===")

    with open(REF_LIST, "r", encoding="utf-8") as f:
        wanted_names = {Path(line.strip()).name for line in f if line.strip()}

    with urlopen(LIST_URL, timeout=60) as resp:
        url_list = [line.decode().strip() for line in resp if line.strip()]

    matching_urls = [u for u in url_list if Path(u).name in wanted_names]
    print(f"Found {len(matching_urls)} matching REAL files to download.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        results = list(
            tqdm(
                ex.map(lambda u: download_file(u, REAL_DIR), matching_urls),
                total=len(matching_urls),
                desc="Downloading REAL",
                unit="file",
            )
        )

    success = sum(results)
    print(f"Downloaded {success}/{len(matching_urls)} REAL files into {REAL_DIR.resolve()}")

def extract_tar(tar_path, out_root):
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=out_root)
    print(f"Extracted {tar_path.name} into {out_root}")

def download_fake():
    print("\n=== DOWNLOADING FAKE VIDEOS (VidProM) ===")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        local_root = snapshot_download(
            repo_id=VIDPROM_REPO,
            repo_type="dataset",
            allow_patterns=["example/*"],
            local_dir=tmp_path,
            local_dir_use_symlinks=False,
        )

        example_dir = Path(local_root) / "example"
        tar_files = glob.glob(str(example_dir / "*.tar"))

        print(f"Found {len(tar_files)} VidProM .tar files to extract.")
        FAKE_DIR.mkdir(parents=True, exist_ok=True)

        for tar_path in map(Path, tar_files):
            extract_tar(tar_path, FAKE_DIR)

    print(f"All VidProM example videos extracted into {FAKE_DIR.resolve()}")


def main():
    TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
    download_real()
    download_fake()
    print("\nAll downloads complete.")
    print(f"REAL → {REAL_DIR.resolve()}")
    print(f"FAKE → {FAKE_DIR.resolve()}")


if __name__ == "__main__":
    main()
