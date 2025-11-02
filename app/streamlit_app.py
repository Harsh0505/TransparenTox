import json
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="TransparentOx", page_icon="🧪", layout="centered")

# locate best model pointer
def load_pointer():
    p = Path("app/best_model.json")
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return None

# load model/tokenizer
@st.cache_resource
def load_backend():
    ptr = load_pointer()
    note = ""
    if ptr and Path(ptr["model_dir"]).exists():
        model_id = ptr["model_dir"]
        default_threshold = float(ptr.get("best_threshold", 0.5))
        note = f"Loaded best run: {ptr.get('run_name','(unknown)')}  |  metric={ptr.get('metric')}  score={ptr.get('score'):.3f}"
    else:
        # Fallbacks if pointer missing
        candidates = [Path("models/bert-base"), Path("models/distilbert-base")]
        model_id = None
        for c in candidates:
            if c.exists():
                model_id = str(c)
                break
        if model_id is None:
            model_id = "distilbert-base-uncased"  # last-resort remote fallback
            note = "Pointer missing; using remote fallback."
        default_threshold = 0.5

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
    mdl.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    return tok, mdl, default_threshold, note, device

tok, mdl, default_threshold, note, device = load_backend()
st.caption(note)

if st.button("Reload best model"):
    load_backend.clear()   # bust @st.cache_resource
    st.experimental_rerun()


st.title("TransparenTox: Toxicity Classifier")
st.write("Enter a comment and adjust the decision threshold if needed.")

text = st.text_area("Comment", height=160, placeholder="Type or paste text here...")
th = st.slider("Decision threshold (toxic if P>=threshold)", 0.0, 1.0, float(default_threshold), 0.01)

@st.cache_data(show_spinner=False)
def predict_proba_batch(texts, max_length=256):
    if not isinstance(texts, list):
        texts = [texts]
    enc = tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = mdl(**enc)
        probs = torch.softmax(out.logits, dim=-1)[:, 1].detach().cpu().numpy().tolist()
    return probs

col1, col2 = st.columns([1,1])
with col1:
    run_btn = st.button("Predict", type="primary")
with col2:
    st.write("")

if run_btn and text.strip():
    prob = predict_proba_batch([text])[0]
    pred = "🛑 Toxic" if prob >= th else "✅ Non-toxic"
    st.markdown(f"**Probability (toxic):** `{prob:.3f}`")
    st.markdown(f"**Prediction:** {pred}")

    with st.expander("What does the threshold do?"):
        st.write(
            "Lower threshold: more recall (catch more toxic) but lower precision. "
            "Higher threshold: higher precision (fewer false positives) but lower recall."
        )

st.markdown("---")
st.caption("Model outputs depend on training data and threshold. Use responsibly.")
