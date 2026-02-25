import argparse
import zipfile
from pathlib import Path

import requests

ZENODO_ID = "18767617"
ZENODO_BASE_URL = f"https://zenodo.org/record/{ZENODO_ID}/files"
ZIP_FILENAME = "model_weights.zip"
ZENODO_URL = f"https://zenodo.org/records/{ZENODO_ID}/files/{ZIP_FILENAME}?download=1"


def download(dest: Path):
    print(f"Downloading from {ZENODO_URL} → {dest} ...")
    r = requests.get(ZENODO_URL, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    print("Download complete.")


def extract_zip(zip_path: Path, out_dir: Path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    print(f"Extracted {zip_path} → {out_dir}")
    zip_path.unlink()


def main():
    zip_path = Path(ZIP_FILENAME)

    download(zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(Path("."))

    zip_path.unlink()
    print("Done. model_weights directory is ready.")


if __name__ == "__main__":
    main()