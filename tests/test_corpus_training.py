from pathlib import Path

from origin.tokenizer import SubwordTokenizer, load_tokenizer
from origin.training.data import prepare_corpus, tokenize_to_cache


def test_bpe_tokenizer_persists_and_round_trips(tmp_path: Path):
    corpus = tmp_path / "train.txt"
    corpus.write_text("Causal language models predict the next token.\n" * 40, encoding="utf-8")
    tokenizer = SubwordTokenizer.train(corpus, vocab_size=256)
    destination = tmp_path / "tokenizer.json"
    tokenizer.save(destination)
    restored = load_tokenizer(destination)
    text = "Causal language models"
    assert restored.decode(restored.encode(text)) == text
    assert restored.eos_id != restored.pad_id


def test_corpus_split_deduplicates_and_cache_reuses(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / "notes.txt"
    source.write_text("alpha beta gamma delta\n" * 20 + "unique train passage\n" * 20, encoding="utf-8")
    manifest = prepare_corpus(source, tmp_path / "prepared", validation_fraction=0.25, seed=7)
    assert manifest["train_lines"] + manifest["validation_lines"] == 2
    assert manifest["duplicates"] == 38
    tokenizer = SubwordTokenizer.train(manifest["train_path"], vocab_size=256)
    cache_path = tmp_path / "cache" / "tokens.bin"
    first = tokenize_to_cache(tokenizer, manifest["train_path"], cache_path, manifest["fingerprint"])
    assert len(first) > 2
    second = tokenize_to_cache(tokenizer, manifest["train_path"], cache_path, manifest["fingerprint"])
    assert list(first) == list(second)
