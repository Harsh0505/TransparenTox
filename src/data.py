from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from .utils import seed_all

def load_toxic_dataset(name="civil_comments", text_col="text", label_col="toxic", model_name="bert-base-uncased", max_length=256):
    seed_all(42)
    if name == "civil_comments":
        ds = load_dataset("civil_comments")
        # binarize label: toxicity >= 0.5 -> 1 else 0
        def _binarize(ex):
            ex[label_col] = int(ex["toxicity"] >= 0.5)
            ex[text_col] = ex["text"]
            return ex
        ds = ds.map(_binarize, remove_columns=[c for c in ds["train"].column_names if c not in [text_col, label_col]])
    elif name == "jigsaw":
        ds = load_dataset("jigsaw_toxicity_pred")  
        text_col, label_col = "comment_text", "toxic"
    else:
        raise ValueError("Unknown dataset")

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    def _tokenize(ex):
        return tok(ex[text_col], truncation=True, max_length=max_length)
    ds = ds.map(_tokenize, batched=True)
    ds = ds.remove_columns([text_col]).rename_column(label_col, "labels")
    ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
    return ds
