from pathlib import Path
import random
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import unicodedata

import numpy as np

MAX_DATA_BYTES = 25 * 1024 * 1024
MAX_CORPUS_BYTES = 5 * 1024 * 1024 * 1024
SUPPORTED_CORPUS_SUFFIXES = {".txt", ".md", ".jsonl", ".tokens"}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "checkpoints", "processed", "outputs"}

def discover_text_files(source: str | Path, output_dir: str | Path | None = None, max_bytes: int = MAX_CORPUS_BYTES) -> list[Path]:
    root = Path(source).expanduser()
    if not root.exists(): raise FileNotFoundError(f"Dataset path not found: {root}")
    if root.is_file():
        files = [root] if root.suffix.lower() in SUPPORTED_CORPUS_SUFFIXES else []
    elif root.is_dir():
        excluded = Path(output_dir).resolve() if output_dir else None
        files = []
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_CORPUS_SUFFIXES or path.name.casefold() == "readme.md": continue
            relative = path.relative_to(root)
            if any(part in SKIP_DIRS or part.startswith(".") for part in relative.parts[:-1]): continue
            if excluded and path.resolve().is_relative_to(excluded): continue
            files.append(path)
        files.sort(key=lambda item: str(item).casefold())
    else:
        raise ValueError("Dataset path must be a regular file or directory.")
    if not files: raise ValueError("No .txt, .md, .jsonl, or .tokens files found in the dataset path.")
    total = sum(path.stat().st_size for path in files)
    if total > max_bytes: raise ValueError(f"Dataset exceeds the {max_bytes / (1024 ** 3):.1f} GiB configured limit.")
    return files

def _records(path: Path):
    if path.suffix.lower() != ".jsonl":
        with path.open("r", encoding="utf-8-sig") as stream:
            yield from stream
        return
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                yield None
                continue
            if isinstance(record, dict) and isinstance(record.get("text"), str):
                yield record["text"]

def prepare_corpus(source: str | Path, output_dir: str | Path, *, validation_fraction: float = 0.02, seed: int = 42, max_bytes: int = MAX_CORPUS_BYTES) -> dict:
    if not 0 <= validation_fraction < 0.5: raise ValueError("validation_fraction must be in [0, 0.5).")
    output = Path(output_dir).resolve()
    files = discover_text_files(source, output, max_bytes)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256()
    counts = {"train_lines": 0, "validation_lines": 0, "duplicates": 0, "invalid_json_lines": 0}
    with tempfile.TemporaryDirectory(prefix="origin-corpus-") as temp_name:
        temp = Path(temp_name)
        db = sqlite3.connect(temp / "dedupe.sqlite3")
        db.execute("CREATE TABLE seen (digest TEXT PRIMARY KEY)")
        train_tmp, val_tmp = temp / "train.txt", temp / "validation.txt"
        with train_tmp.open("w", encoding="utf-8", newline="\n") as train_out, val_tmp.open("w", encoding="utf-8", newline="\n") as val_out:
            for path in files:
                for raw in _records(path):
                    if raw is None:
                        counts["invalid_json_lines"] += 1
                        continue
                    text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", raw)).strip()
                    if len(text) < 2: continue
                    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    if not db.execute("INSERT OR IGNORE INTO seen VALUES (?)", (digest,)).rowcount:
                        counts["duplicates"] += 1
                        continue
                    fingerprint.update(digest.encode("ascii")); fingerprint.update(b"\n")
                    bucket = int(hashlib.sha256(f"{seed}:{digest}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                    if validation_fraction and bucket < validation_fraction:
                        val_out.write(text + "\n"); counts["validation_lines"] += 1
                    else:
                        train_out.write(text + "\n"); counts["train_lines"] += 1
        db.close()
        if counts["train_lines"] == 0: raise ValueError("No usable text records found. JSONL files must contain a string 'text' field.")
        train_path, val_path = output / "train.txt", output / "validation.txt"
        os.replace(train_tmp, train_path)
        if counts["validation_lines"]:
            os.replace(val_tmp, val_path)
        else:
            val_path.unlink(missing_ok=True)
    return {**counts, "source_files": [str(path) for path in files], "source_bytes": sum(path.stat().st_size for path in files), "fingerprint": fingerprint.hexdigest(), "train_path": str(train_path), "validation_path": str(val_path) if counts["validation_lines"] else None}

def tokenize_to_cache(tokenizer, text_path: str | Path, binary_path: str | Path, corpus_fingerprint: str):
    import hashlib
    target = Path(binary_path); target.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = target.with_suffix(target.suffix + ".json")
    tokenizer_id = hashlib.sha256(tokenizer.model_proto).hexdigest()
    text_digest = hashlib.sha256()
    with Path(text_path).open("rb") as source_stream:
        for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
            text_digest.update(chunk)
    text_fingerprint = text_digest.hexdigest()
    if target.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected_bytes = int(metadata["num_tokens"]) * np.dtype(np.uint32).itemsize
            if metadata.get("tokenizer_id") == tokenizer_id and metadata.get("corpus_fingerprint") == corpus_fingerprint and metadata.get("text_fingerprint") == text_fingerprint and target.stat().st_size == expected_bytes:
                return np.memmap(target, dtype=np.uint32, mode="r", shape=(metadata["num_tokens"],))
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    def encoded_lines():
        with Path(text_path).open("r", encoding="utf-8") as stream:
            for line in stream:
                ids = tokenizer.encode(line.rstrip("\n"), add_eos=True)
                if ids: yield ids
    count = sum(len(ids) for ids in encoded_lines())
    if count <= 1: raise ValueError("Dataset has too few tokens after tokenization.")
    with tempfile.NamedTemporaryFile(prefix=target.stem, suffix=".tmp", dir=target.parent, delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        cache = np.memmap(temp_path, dtype=np.uint32, mode="w+", shape=(count,))
        position = 0
        for ids in encoded_lines():
            cache[position:position + len(ids)] = ids
            position += len(ids)
        cache.flush(); del cache
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)
    metadata = {"num_tokens": count, "tokenizer_id": tokenizer_id, "corpus_fingerprint": corpus_fingerprint, "text_fingerprint": text_fingerprint}
    metadata_tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_tmp.write_text(json.dumps(metadata), encoding="utf-8")
    os.replace(metadata_tmp, metadata_path)
    return np.memmap(target, dtype=np.uint32, mode="r", shape=(count,))

def load_texts(path: str | Path) -> list[str]:
    source = Path(path)
    if not source.exists() or not source.is_file(): raise FileNotFoundError(f"Dataset file not found: {source}")
    if source.stat().st_size > MAX_DATA_BYTES: raise ValueError("Dataset exceeds the 25 MB safety limit.")
    texts = [line.strip() for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not texts: raise ValueError("Dataset contains no non-empty text lines.")
    return list(dict.fromkeys(texts))

def split_texts(texts: list[str], validation_fraction: float = 0.1, seed: int = 42):
    if not 0 <= validation_fraction < 1 or not texts: raise ValueError("Invalid split fraction or empty dataset.")
    values = list(texts); random.Random(seed).shuffle(values)
    n_val = min(len(values) - 1, int(len(values) * validation_fraction))
    return values[n_val:], values[:n_val]
