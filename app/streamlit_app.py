import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

@st.cache_resource
def load_model(path="models/bert-base"):
    tok = AutoTokenizer.from_pretrained(path)
    mdl = AutoModelForSequenceClassification.from_pretrained(path)
    mdl.eval()
    return tok, mdl

st.title("TransparenTox: Explainable Toxic Comment Detector")
user_text = st.text_area("Enter text:", height=150)
if st.button("Predict") and user_text.strip():
    tok, mdl = load_model()
    enc = tok(user_text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        out = mdl(**enc)
        prob = torch.softmax(out.logits, dim=-1)[0].tolist()[1]
    st.write(f"**Toxicity probability:** {prob:.3f}")
    st.caption("Explanations coming soon (SHAP).")


