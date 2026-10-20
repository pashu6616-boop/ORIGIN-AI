# ORIGIN AI v0.1 architecture

ORIGIN v0.1 is text-only and experimental. `CharacterTokenizer` builds a deterministic character vocabulary from the training corpus. `DecoderTransformer` uses learned token and position embeddings, pre-normalized causal self-attention, feed-forward residual blocks, and next-character cross-entropy. All weights start randomly; there is no pretrained model download or provider API.

The training package loads UTF-8 line corpora, removes exact duplicate lines, splits lines deterministically, samples next-token batches, reports train and validation loss, and saves model and optimizer state. Checkpoints include separate tokenizer JSON and metadata JSON beside `.pt` weights. Only load checkpoint files you created or trust because PyTorch checkpoint files use Python serialization.

The knowledge module is independent of the language model. It chunks local TXT/Markdown and ranks chunks by query term frequency. The app displays excerpts verbatim with the filename. It does not ask the model to synthesize document answers or create citations. Multimodal message/context records leave room for future adapters; image, audio, and speech output return explicit unsupported status.

## Resource expectations

The shipped model is 1 layer, 32 hidden dimensions, 2 heads, and 64 character context. Its parameter storage is well under 1 MB for ordinary text vocabularies; training also needs activations and optimizer state, so RAM varies with vocabulary, batch size, and context. The four-step CPU smoke run is intentionally tiny. It verifies the pipeline, not language quality. Useful model quality would require a large authorized corpus and much more compute.
