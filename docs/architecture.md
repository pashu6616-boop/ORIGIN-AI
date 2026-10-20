# ORIGIN AI v0.1 architecture

ORIGIN v0.1 is text-only and experimental. A locally trained SentencePiece BPE tokenizer builds subword units from the training corpus; no tokenizer or model is downloaded. `DecoderTransformer` uses learned token and position embeddings, pre-normalized causal self-attention, feed-forward residual blocks, and next-token cross-entropy. All model weights start randomly; there is no pretrained model download or provider API.

The training package streams UTF-8 TXT, Markdown, SentencePiece `.tokens`, and JSONL text records. It normalizes Unicode and whitespace, removes exact duplicates through a temporary SQLite index, makes a deterministic holdout split, trains BPE only on training text, and writes token IDs to memory-mapped caches. Batches are read from those caches instead of loading the whole corpus into RAM. Separate official validation data can be supplied. Periodic checkpoints save optimizer state, tokenizer, config, metrics, and step so long runs can resume. Only load checkpoint files you created or trust because PyTorch checkpoint files use Python serialization.

The knowledge module is independent of the language model. It chunks local TXT/Markdown and ranks chunks by query term frequency. The app displays excerpts verbatim with the filename. It does not ask the model to synthesize document answers or create citations. Multimodal message/context records leave room for future adapters; image, audio, and speech output return explicit unsupported status.

## Resource expectations

The default model is 2 layers, 96 hidden dimensions, 4 heads, and 128 subword tokens of context. The parameter and Adam optimizer footprint is estimated in the training report; activations and tokenizer/data caches add RAM and disk use. Batch size, context, width, vocabulary, and step count are configurable. This remains a small experimental model. WikiText-2 supports a reproducible learning experiment but does not establish general-purpose assistant quality. Larger authorized corpora and much more compute are required for stronger results.
