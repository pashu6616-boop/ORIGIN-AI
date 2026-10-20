from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

def main():
    import streamlit as st
    import torch
    from origin.config import ModelConfig
    from origin.knowledge import DocumentIndex
    from origin.model import DecoderTransformer
    from origin.tokenizer import CharacterTokenizer, load_tokenizer
    from origin.training.checkpoint import load_checkpoint
    st.set_page_config(page_title="ORIGIN AI v0.1", page_icon="◉", layout="centered")
    st.title("◉ ORIGIN AI"); st.caption("v0.1 · A small, experimental language-model foundation")
    checkpoint = ROOT / "checkpoints" / "latest.pt"
    loaded = False
    if checkpoint.exists() and checkpoint.with_suffix(".tokenizer.json").exists():
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            tokenizer = load_tokenizer(checkpoint.with_suffix(".tokenizer.json"))
            model = DecoderTransformer(ModelConfig(**payload["model_config"]))
            load_checkpoint(checkpoint, model)
            st.success(f"Native model: trained checkpoint, step {payload.get('step', '?')} · CPU")
            loaded = True
        except (OSError, RuntimeError, KeyError, ValueError, TypeError):
            st.error("The local checkpoint could not be loaded. Start a fresh training run to create a replacement checkpoint.")
    if not loaded:
        tokenizer = CharacterTokenizer.from_texts(["ORIGIN AI is an experimental language model. "])
        model = DecoderTransformer(ModelConfig(vocab_size=tokenizer.vocab_size, block_size=64, n_embd=32, n_head=2, n_layer=1))
        st.warning("Native model: random initialization, not trained. Generated text will not be useful yet. Run the CPU smoke test for a tiny demonstration checkpoint.")
    if "messages" not in st.session_state: st.session_state.messages = []
    if st.button("Clear conversation"):
        st.session_state.messages = []; st.rerun()
    mode = st.radio("Answer source", ["Native experimental model", "Search supplied documents"], horizontal=True)
    uploaded = st.file_uploader("Optional local evidence (TXT or Markdown)", type=["txt", "md"], accept_multiple_files=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.text(message["content"])
    prompt = st.chat_input("Ask ORIGIN (up to 500 characters)")
    if prompt:
        if len(prompt) > 500: st.error("Please keep the prompt to 500 characters or fewer."); return
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"): st.text(prompt)
        with st.chat_message("assistant"):
            if mode == "Search supplied documents":
                if len(uploaded or []) > 10 or sum(file.size for file in (uploaded or [])) > 10 * 1024 * 1024:
                    st.error("Please upload at most 10 files and 10 MB total."); return
                index = DocumentIndex()
                for file in uploaded or []:
                    if file.size > 5 * 1024 * 1024: continue
                    index.add_text(file.name, file.getvalue().decode("utf-8", errors="replace"))
                results = index.search(prompt)
                if not results:
                    answer = "No relevant excerpt was found in the supplied documents."; st.write(answer)
                else:
                    st.markdown("**Retrieved evidence (shown verbatim; not learned by the model):**")
                    for item in results:
                        st.caption(f"Source: {item.source}"); st.code(item.excerpt, language=None)
                    answer = "\n\n".join(f"Source: {item.source}\n{item.excerpt}" for item in results)
            else:
                prompt_ids = tokenizer.encode(prompt[-500:])
                if not prompt_ids: answer = "Enter at least one supported character."
                else:
                    tokens = torch.tensor([prompt_ids[-model.config.block_size:]], dtype=torch.long)
                    generated = model.generate(tokens, max_new_tokens=120, temperature=0.9, top_k=20, eos_token_id=getattr(tokenizer, "eos_id", None))
                    answer = tokenizer.decode(generated[0, len(tokens[0]):].tolist()) or "(No new characters generated.)"
                st.text(answer)
        st.session_state.messages.append({"role":"assistant","content":answer})
    st.caption("Native generation is experimental. Retrieval only displays matching excerpts; it does not train or update the model.")

if __name__ == "__main__": main()
