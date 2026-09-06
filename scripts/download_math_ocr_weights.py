"""Download the official Pix2Tex checkpoints for local-only math transcription.

Run this script once during setup. It never runs as part of the API or document
ingestion path, so uploads and queries remain fully local after setup.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen


RELEASE_BASE_URL = "https://github.com/lukas-blecher/LaTeX-OCR/releases/download/v0.0.1"
FILES = ("weights.pth", "image_resizer.pth")


def download(url: str, destination: Path) -> str:
    temporary = destination.with_suffix(f"{destination.suffix}.partial")
    digest = hashlib.sha256()
    with urlopen(url) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
    temporary.replace(destination)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download trusted local Pix2Tex checkpoints.")
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("artifacts/math_ocr/checkpoints"),
        help="Directory used by MATH_OCR_CHECKPOINT (default: artifacts/math_ocr/checkpoints).",
    )
    arguments = parser.parse_args()
    arguments.directory.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        destination = arguments.directory / filename
        if destination.is_file() and destination.stat().st_size > 0:
            print(f"Keeping existing {destination}")
            continue
        checksum = download(f"{RELEASE_BASE_URL}/{filename}", destination)
        print(f"Downloaded {destination} (SHA-256: {checksum})")


if __name__ == "__main__":
    main()
