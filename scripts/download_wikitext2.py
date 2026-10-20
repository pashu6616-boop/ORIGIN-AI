"""Optional small, openly licensed real-text corpus for a local training run."""
from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import shutil
import tempfile
from urllib.request import Request, urlopen

BASE_URL = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-v1"
MAX_FILE_BYTES = 10 * 1024 * 1024
FILES = {
    "wiki.train.tokens": ("train-00000-of-00001.parquet", "dfc27e4360c639dc1fba1e403bfffd53af4a5c75d5363b5724d49bf12d07cce6"),
    "wiki.valid.tokens": ("validation-00000-of-00001.parquet", "717de9a0c1c0b0b1dfdd8f1e6ad8a30ece618bbde81f5da8207277547d324215"),
    "wiki.test.tokens": ("test-00000-of-00001.parquet", "e6b3913da714b63a60a571698b20ff15441fb015783ea1b5285f707d4f2f00a9"),
}

def main() -> None:
    parser = argparse.ArgumentParser(description="Download the optional Salesforce WikiText-2 corpus; no model or credentials are downloaded.")
    parser.add_argument("--destination", default="data/wikitext-2", help="New local directory for the dataset files.")
    args = parser.parse_args()
    destination = Path(args.destination).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        if all((destination / name).is_file() for name in FILES):
            print(f"WikiText-2 is already present at {destination}")
            return
        parser.error(f"Destination contains files already; nothing was overwritten: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="origin-wikitext2-") as temp_name:
            staged = Path(temp_name) / "dataset"
            staged.mkdir()
            for basename, (source_name, expected_sha256) in FILES.items():
                url = f"{BASE_URL}/{source_name}?download=true"
                file_request = Request(url, headers={"User-Agent": "ORIGIN-AI/0.1 dataset downloader"})
                target = staged / f"{basename}.parquet"
                digest = hashlib.sha256()
                received = 0
                with urlopen(file_request, timeout=60) as response, target.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        received += len(chunk)
                        if received > MAX_FILE_BYTES: raise ValueError(f"Dataset file exceeds the 10 MiB limit: {basename}")
                        digest.update(chunk); output.write(chunk)
                if digest.hexdigest() != expected_sha256:
                    raise ValueError(f"SHA-256 validation failed for {basename}.")
            # Current Hugging Face representation is Parquet; decode text using the
            # repository's declared dependency rather than adding a data runtime.
            try:
                import pyarrow.parquet as parquet
            except ImportError as exc:
                raise RuntimeError("Install pyarrow to convert the maintained WikiText files: python -m pip install pyarrow") from exc
            for basename in FILES:
                parquet_path = staged / f"{basename}.parquet"
                table = parquet.read_table(parquet_path, columns=["text"])
                with (staged / basename).open("w", encoding="utf-8", newline="\n") as output:
                    for row in table.column("text").to_pylist():
                        output.write((row or "") + "\n")
            (staged / "SOURCE.txt").write_text(
                "WikiText-2 word-level data from Salesforce AI Research.\n"
                "Source: https://huggingface.co/datasets/Salesforce/wikitext/tree/main/wikitext-2-v1\n"
                "License: Creative Commons Attribution-ShareAlike. Review the source terms before redistribution.\n"
                "The test split is preserved here and is not used by ORIGIN's training command.\n",
                encoding="utf-8",
            )
            for file in staged.glob("wiki.*.tokens"):
                file.replace(destination / file.name)
            (staged / "SOURCE.txt").replace(destination / "SOURCE.txt")
    except Exception as exc:
        parser.error(f"Could not download or validate WikiText-2: {exc}")
    print(f"Downloaded train, validation, and test splits to {destination}")

if __name__ == "__main__": main()
