import json
from pathlib import Path
import pytest
import torch
from origin.config import ModelConfig
from origin.knowledge import DocumentIndex
from origin.model import DecoderTransformer
from origin.multimodal import CapabilityStatus, audio_input, image_input, text_to_speech
from origin.tokenizer import CharacterTokenizer
from origin.training.checkpoint import load_checkpoint, save_checkpoint
from origin.training.data import load_texts
from origin.training.trainer import train_steps

def small_model(vocab_size=8):
    return DecoderTransformer(ModelConfig(vocab_size=vocab_size, block_size=8, n_embd=8, n_head=2, n_layer=1))

def test_tokenizer_round_trip_and_unknown():
    tokenizer = CharacterTokenizer.from_texts(["cat"])
    assert tokenizer.decode(tokenizer.encode("cat")) == "cat"
    assert tokenizer.encode("z")[0] == 1

def test_causal_attention_masks_future_tokens():
    torch.manual_seed(3); model = small_model().eval()
    first, _ = model(torch.tensor([[1,2,3]])); second, _ = model(torch.tensor([[1,2,6]]))
    assert torch.allclose(first[:, :2], second[:, :2], atol=1e-6)

def test_forward_shapes_and_loss():
    logits, loss = small_model()(torch.randint(0,8,(2,5)), torch.randint(0,8,(2,5)))
    assert logits.shape == (2,5,8) and loss is not None and torch.isfinite(loss)

def test_invalid_config():
    with pytest.raises(ValueError): ModelConfig(vocab_size=4,n_embd=7,n_head=2).validate()

def test_checkpoint_save_load_and_resume(tmp_path):
    tok = CharacterTokenizer.from_texts(["some training text"])
    model = DecoderTransformer(ModelConfig(vocab_size=tok.vocab_size,block_size=8,n_embd=8,n_head=2,n_layer=1))
    opt = torch.optim.AdamW(model.parameters()); target = tmp_path / "model.pt"
    save_checkpoint(target,model,opt,tok,{"steps":1},{"loss":1.0},1)
    restored = DecoderTransformer(model.config); opt2 = torch.optim.AdamW(restored.parameters())
    assert load_checkpoint(target,restored,opt2)["step"] == 1
    assert CharacterTokenizer.load(target.with_suffix(".tokenizer.json")).vocabulary == tok.vocabulary
    assert json.loads(target.with_suffix(".metadata.json").read_text())["step"] == 1

def test_generation_length_and_bad_settings():
    assert small_model().generate(torch.tensor([[1,2]]),max_new_tokens=3).shape[1] == 5
    with pytest.raises(ValueError): small_model().generate(torch.tensor([[1]]),max_new_tokens=-1)

def test_cpu_training_step():
    model=small_model(vocab_size=16); opt=torch.optim.AdamW(model.parameters())
    metrics=train_steps(model,opt,list(range(16))*4,[],steps=1,block_size=8,batch_size=2)
    assert len(metrics["train_loss"]) == len(metrics["validation_loss"]) == 1

def test_document_ingestion_and_retrieval(tmp_path):
    file=tmp_path/"notes.md"; file.write_text("Causal attention uses a triangular mask. Retrieval returns relevant evidence.")
    index=DocumentIndex(chunk_words=20,overlap_words=2)
    assert index.add_files([file]) > 0 and index.search("triangular attention mask")[0].source == "notes.md"
    assert not index.search("quasar")

def test_missing_dataset_and_invalid_chunk_config(tmp_path):
    with pytest.raises(FileNotFoundError): load_texts(tmp_path/"missing.txt")
    with pytest.raises(ValueError): DocumentIndex(chunk_words=2,overlap_words=2)

def test_duplicate_and_empty_dataset(tmp_path):
    path=tmp_path/"data.txt"; path.write_text("hello\nhello\nworld\n")
    assert load_texts(path)==["hello","world"]
    path.write_text("\n \n")
    with pytest.raises(ValueError): load_texts(path)

def test_multimodal_explicitly_unsupported():
    for result in (image_input(None),audio_input(None),text_to_speech("hi")):
        assert result.status == CapabilityStatus.UNSUPPORTED and "not implemented" in result.message

def test_chat_startup_smoke():
    from streamlit.testing.v1 import AppTest
    app_path=Path(__file__).resolve().parents[1]/"app"/"main.py"
    result=AppTest.from_file(str(app_path)).run(timeout=30)
    assert not result.exception
