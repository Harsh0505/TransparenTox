# src/train.py
import argparse
import json
from pathlib import Path

import evaluate
import numpy as np
import yaml
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    average_precision_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    AutoTokenizer,
)

from .model import build_model
from .data import load_toxic_dataset
from .utils import ensure_dirs

# Load Metrics
_metric_acc = evaluate.load("accuracy")
_metric_prec_bin = evaluate.load("precision")
_metric_rec_bin = evaluate.load("recall")
_metric_f1_bin = evaluate.load("f1")
_metric_f1_macro = evaluate.load("f1")
try:
    _metric_roc_auc = evaluate.load("roc_auc")
except Exception:
    _metric_roc_auc = None  # fall back to sklearn.roc_auc_score


def _sanitise_config(cfg: dict) -> dict:
    """Ensure numeric fields are proper types (float/int) even if YAML/CLI provided strings."""
    def to_float(x, default=None):
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        return float(x)

    def to_int(x, default=None):
        if x is None:
            return default
        if isinstance(x, int):
            return x
        return int(x)

    cfg["lr"] = to_float(cfg.get("lr", 2e-5), 2e-5)
    cfg["weight_decay"] = to_float(cfg.get("weight_decay", 0.01), 0.01)
    cfg["train_batch_size"] = to_int(cfg.get("train_batch_size", 16), 16)
    cfg["eval_batch_size"] = to_int(cfg.get("eval_batch_size", 32), 32)
    cfg["num_epochs"] = to_int(cfg.get("num_epochs", 1), 1)
    for k in ["train_subset", "eval_subset", "test_subset"]:
        if k in cfg and cfg[k] is not None:
            cfg[k] = to_int(cfg[k])
    return cfg


def compute_metrics_builder():
    """Create metrics func that works on old/new evaluate and returns rich metrics."""
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        # positive-class probabilities
        probs_pos = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

        out = {
            "accuracy": _metric_acc.compute(predictions=preds, references=labels)["accuracy"],
            "precision": _metric_prec_bin.compute(predictions=preds, references=labels, average="binary", pos_label=1)["precision"],
            "recall": _metric_rec_bin.compute(predictions=preds, references=labels, average="binary", pos_label=1)["recall"],
            "f1": _metric_f1_bin.compute(predictions=preds, references=labels, average="binary")["f1"],
            "f1_macro": _metric_f1_macro.compute(predictions=preds, references=labels, average="macro")["f1"],
            "pr_auc_ap": float(average_precision_score(labels, probs_pos)),
        }
        # ROC-AUC via evaluate if available; else sklearn fallback
        try:
            if _metric_roc_auc is not None:
                out["roc_auc"] = _metric_roc_auc.compute(prediction_scores=probs_pos, references=labels)["roc_auc"]
            else:
                out["roc_auc"] = float(roc_auc_score(labels, probs_pos))
        except Exception:
            pass

        return out

    return compute_metrics


class WeightedTrainer(Trainer):
    """
    Trainer that applies class weights in CrossEntropyLoss to handle imbalance.
    Compatible with Trainer APIs that pass extra kwargs like `num_items_in_batch`.
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits

        if labels.dtype != torch.long:
            labels = labels.long()
        labels = labels.view(-1)

        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def _make_training_args(cfg):
    """
    Build TrainingArguments from config dict, handling version differences.
    """
    base_kwargs = dict(
        output_dir=cfg["output_dir"],
        learning_rate=cfg["lr"],
        per_device_train_batch_size=cfg["train_batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        num_train_epochs=cfg["num_epochs"],
        weight_decay=cfg["weight_decay"],
        logging_steps=50,
        dataloader_num_workers=4,
    )
    try:
        return TrainingArguments(
            **base_kwargs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to=[],  # disable default wandb/tensorboard logging
        )
    except TypeError:
        return TrainingArguments(**base_kwargs)


def _best_threshold_from_validation(trainer, eval_ds):
    """
    Grid-search probability threshold on validation to maximize binary F1.
    Saves the chosen threshold to reports/best_threshold.txt
    """
    out = trainer.predict(eval_ds)
    logits = out.predictions
    labels = np.array(eval_ds["labels"])
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

    ts = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, 0.0
    for t in ts:
        preds = (probs >= t).astype(int)
        f1 = _metric_f1_bin.compute(predictions=preds, references=labels, average="binary")["f1"]
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    Path("reports").mkdir(exist_ok=True)
    with open("reports/best_threshold.txt", "w") as f:
        f.write(str(best_t))
    return best_t, best_f1


def train(config):
    """
    Train + evaluate. Save:
      - models/<name>/
      - reports/metrics.json
      - reports/confusion_matrix.png
      - reports/classification_report.txt
      - reports/best_threshold.txt
    """
    ensure_dirs()
    config = _sanitise_config(config)

    # load dataset
    ds = load_toxic_dataset(
        name=config["dataset"],
        text_col=config["text_col"],
        label_col=config["label_col"],
        model_name=config["model_name"],
        max_length=config["max_length"],
    )

    # split into train, validation and test sets
    train_ds = ds["train"].shuffle(seed=42).select(range(config.get("train_subset", 6000)))
    eval_ds = ds["validation"].select(range(config.get("eval_subset", 2000)))
    test_ds = ds["test"].select(range(config.get("test_subset", 2000)))

    # model and tokeniser/collator
    model = build_model(config["model_name"])
    tok = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
    collator = DataCollatorWithPadding(tokenizer=tok)

    # class weights to handle dataset imbalance (92% non-toxic in civil_comments)
    y_train = np.array(train_ds["labels"])
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float)

    # training args and metrics
    args = _make_training_args(config)
    compute_metrics = compute_metrics_builder()

    # trainer
    try:
        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tok,
            data_collator=collator,
            compute_metrics=compute_metrics,
            class_weights=class_weights,
        )
    except TypeError:
        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tok,
            data_collator=collator,
            compute_metrics=compute_metrics,
            class_weights=class_weights,
        )

    # train
    trainer.train()

    # threshold search on validation for best F1
    best_t, best_val_f1 = _best_threshold_from_validation(trainer, eval_ds)

    # evaluate on test set and compute thresholded combined metrics/report below
    eval_metrics = trainer.evaluate(test_ds)
    combined_metrics = {**eval_metrics, "best_threshold": best_t, "best_val_f1": best_val_f1}
    print(combined_metrics)

    # save model/tokenizer
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    trainer.save_model(config["output_dir"])
    tok.save_pretrained(config["output_dir"])

    # reports
    Path("reports").mkdir(exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump(combined_metrics, f, indent=2)

    # confusion matrix/report using the chosen threshold
    pred_out = trainer.predict(test_ds)
    logits = pred_out.predictions
    y_true = np.array(test_ds["labels"])
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds_thresh = (probs >= best_t).astype(int)

    cm = confusion_matrix(y_true, preds_thresh)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix (Test) @ threshold={best_t:.2f}")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["non-toxic", "toxic"])
    plt.yticks(tick_marks, ["non-toxic", "toxic"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    fig.savefig("reports/confusion_matrix.png", dpi=180)
    plt.close(fig)

    with open("reports/classification_report.txt", "w") as f:
        f.write(f"Threshold used: {best_t:.2f}\n\n")
        f.write(classification_report(y_true, preds_thresh, target_names=["non-toxic", "toxic"]))

    return combined_metrics


def _merge_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.model_name:
        cfg["model_name"] = args.model_name
    if args.epochs is not None:
        cfg["num_epochs"] = args.epochs
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.train_subset is not None:
        cfg["train_subset"] = args.train_subset
    if args.eval_subset is not None:
        cfg["eval_subset"] = args.eval_subset
    if args.test_subset is not None:
        cfg["test_subset"] = args.test_subset
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TransparentOx training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model_name", type=str, help="HF model name (e.g., bert-base-uncased)")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--train_subset", type=int, help="Num train examples (subset)")
    parser.add_argument("--eval_subset", type=int, help="Num eval examples (subset)")
    parser.add_argument("--test_subset", type=int, help="Num test examples (subset)")
    parser.add_argument("--output_dir", type=str, help="Where to save model")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = _merge_overrides(cfg, args)
    train(cfg)
