#!/usr/bin/env python3
# Modified by Cursor: hierarchical multi-head RoBERTa fine-tuning for PAN Task 2.
"""Fully fine-tune RoBERTa-base with taxonomy-aligned heads on an NVIDIA H200."""

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import ClassLabel, load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    RobertaConfig,
    RobertaModel,
    Trainer,
    TrainingArguments,
)

IGNORE_INDEX = -100
FLAT_LABELS = list(range(6))

# Head binary targets (0/1). See README in docstring paths below.
# R0: 0 = pure human (label 0), 1 = machine-involved path (labels 1-5)
# H0: 0 = homogeneous (not 4), 1 = segment-mixed (label 4) — only on R0=1
# H2: 0 = machine-initiated (2,5), 1 = human-initiated (1,3) — homogeneous only
# H3a: 0 = polish (1), 1 = continue (3) — human-initiated only
# H3b: 0 = humanized (2), 1 = human-edited (5) — machine-initiated only


def clean_text(text: str) -> str:
    text = str(text).replace("_SEP_", " ").replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", text).strip()


def flat_label_to_heads(label: int) -> dict[str, int]:
    """Map official flat label 0-5 to per-head supervision (-100 = masked)."""
    label = int(label)
    heads = {
        "r0": IGNORE_INDEX,
        "h0": IGNORE_INDEX,
        "h2": IGNORE_INDEX,
        "h3a": IGNORE_INDEX,
        "h3b": IGNORE_INDEX,
        "leaf_label": label,
    }
    if label == 0:
        heads["r0"] = 0
        return heads
    heads["r0"] = 1
    if label == 4:
        heads["h0"] = 1
        return heads
    heads["h0"] = 0
    if label in (1, 3):
        heads["h2"] = 1
        heads["h3a"] = 0 if label == 1 else 1
        return heads
    if label in (2, 5):
        heads["h2"] = 0
        heads["h3b"] = 0 if label == 2 else 1
        return heads
    raise ValueError(f"Unsupported flat label: {label}")


def decode_leaf_from_head_logits(
    r0_logits: np.ndarray,
    h0_logits: np.ndarray,
    h2_logits: np.ndarray,
    h3a_logits: np.ndarray,
    h3b_logits: np.ndarray,
) -> np.ndarray:
    """Path decoding: pick flat label 0-5 with highest sum of log-probs along the path."""
    r0_log = _log_softmax_np(r0_logits)
    h0_log = _log_softmax_np(h0_logits)
    h2_log = _log_softmax_np(h2_logits)
    h3a_log = _log_softmax_np(h3a_logits)
    h3b_log = _log_softmax_np(h3b_logits)

    batch_size = r0_logits.shape[0]
    scores = np.full((batch_size, 6), -np.inf, dtype=np.float64)

    # Leaf 0
    scores[:, 0] = r0_log[:, 0]
    # Rest branch
    rest = r0_log[:, 1]
    scores[:, 4] = rest + h0_log[:, 1]
    homog = rest + h0_log[:, 0]
    scores[:, 1] = homog + h2_log[:, 1] + h3a_log[:, 0]
    scores[:, 3] = homog + h2_log[:, 1] + h3a_log[:, 1]
    scores[:, 2] = homog + h2_log[:, 0] + h3b_log[:, 0]
    scores[:, 5] = homog + h2_log[:, 0] + h3b_log[:, 1]
    return scores.argmax(axis=-1).astype(np.int64)


def _log_softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return logits - np.log(exp.sum(axis=-1, keepdims=True) + 1e-12)


class HierarchicalRobertaConfig(RobertaConfig):
    model_type = "hierarchical_roberta"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = 5


class HierarchicalRoberta(PreTrainedModel):
    config_class = HierarchicalRobertaConfig

    def __init__(self, config: HierarchicalRobertaConfig):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        hidden = config.hidden_size
        self.r0_head = nn.Linear(hidden, 2)
        self.h0_head = nn.Linear(hidden, 2)
        self.h2_head = nn.Linear(hidden, 2)
        self.h3a_head = nn.Linear(hidden, 2)
        self.h3b_head = nn.Linear(hidden, 2)
        self.head_loss_fn = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self.post_init()

    def _pooled(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        if outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0, :]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        r0=None,
        h0=None,
        h2=None,
        h3a=None,
        h3b=None,
        leaf_label=None,
        labels=None,
        return_dict=True,
        **kwargs,
    ):
        pooled = self._pooled(input_ids, attention_mask)
        r0_logits = self.r0_head(pooled)
        h0_logits = self.h0_head(pooled)
        h2_logits = self.h2_head(pooled)
        h3a_logits = self.h3a_head(pooled)
        h3b_logits = self.h3b_head(pooled)

        leaf_logits = self._leaf_scores_from_logits(
            r0_logits, h0_logits, h2_logits, h3a_logits, h3b_logits
        )

        loss = None
        target_leaf = labels if labels is not None else leaf_label
        if r0 is not None:
            loss = self._weighted_head_loss(
                r0_logits,
                h0_logits,
                h2_logits,
                h3a_logits,
                h3b_logits,
                r0,
                h0,
                h2,
                h3a,
                h3b,
                getattr(self, "head_loss_weights", None),
                getattr(self, "h3b_class_weight", None),
            )

        if not return_dict:
            return (loss, leaf_logits) if loss is not None else (leaf_logits,)

        # Only expose leaf logits: Transformers gathers every *logits key into eval
        # predictions; extra head tensors break compute_metrics (inhomogeneous tuple).
        return {
            "loss": loss,
            "logits": leaf_logits,
        }

    def _leaf_scores_from_logits(self, r0_logits, h0_logits, h2_logits, h3a_logits, h3b_logits):
        """Differentiable leaf scores for Trainer metrics (log-prob sum per path)."""
        r0_log = F.log_softmax(r0_logits, dim=-1)
        h0_log = F.log_softmax(h0_logits, dim=-1)
        h2_log = F.log_softmax(h2_logits, dim=-1)
        h3a_log = F.log_softmax(h3a_logits, dim=-1)
        h3b_log = F.log_softmax(h3b_logits, dim=-1)

        rest = r0_log[:, 1]
        homog = rest + h0_log[:, 0]
        scores = torch.stack(
            [
                r0_log[:, 0],
                homog + h2_log[:, 1] + h3a_log[:, 0],
                homog + h2_log[:, 0] + h3b_log[:, 0],
                homog + h2_log[:, 1] + h3a_log[:, 1],
                rest + h0_log[:, 1],
                homog + h2_log[:, 0] + h3b_log[:, 1],
            ],
            dim=-1,
        )
        return scores

    def _weighted_head_loss(
        self,
        r0_logits,
        h0_logits,
        h2_logits,
        h3a_logits,
        h3b_logits,
        r0,
        h0,
        h2,
        h3a,
        h3b,
        weights: dict[str, float] | None,
        h3b_class_weight: torch.Tensor | None,
    ):
        weights = weights or {}
        w_r0 = weights.get("r0", 1.0)
        w_h0 = weights.get("h0", 1.0)
        w_h2 = weights.get("h2", 1.0)
        w_h3a = weights.get("h3a", 1.0)
        w_h3b = weights.get("h3b", 1.0)

        losses = [
            w_r0 * self.head_loss_fn(r0_logits, r0),
            w_h0 * self.head_loss_fn(h0_logits, h0),
            w_h2 * self.head_loss_fn(h2_logits, h2),
            w_h3a * self.head_loss_fn(h3a_logits, h3a),
        ]
        if h3b_class_weight is not None:
            losses.append(
                w_h3b
                * F.cross_entropy(
                    h3b_logits,
                    h3b,
                    weight=h3b_class_weight,
                    ignore_index=IGNORE_INDEX,
                )
            )
        else:
            losses.append(w_h3b * self.head_loss_fn(h3b_logits, h3b))
        return sum(losses) / len(losses)


@dataclass
class HierarchicalDataCollator(DataCollatorWithPadding):
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        head_keys = ("r0", "h0", "h2", "h3a", "h3b", "leaf_label")
        head_batch = {key: [row.pop(key) for row in features] for key in head_keys}
        batch = super().__call__(features)
        for key in ("r0", "h0", "h2", "h3a", "h3b"):
            batch[key] = torch.tensor(head_batch[key], dtype=torch.long)
        # Trainer eval gathers every *label* field; do not also pass leaf_label.
        batch["labels"] = torch.tensor(head_batch["leaf_label"], dtype=torch.long)
        return batch


def flat_label_predictions_from_logits(logits) -> np.ndarray:
    """Reduce Trainer eval outputs to per-example flat labels 0-5."""
    if isinstance(logits, (tuple, list)):
        arrays = [np.asarray(item) for item in logits]
        for arr in arrays:
            if arr.ndim == 2 and arr.shape[1] == len(FLAT_LABELS):
                return np.argmax(arr, axis=-1)
        binary_heads = [arr for arr in arrays if arr.ndim == 2 and arr.shape[1] == 2]
        if len(binary_heads) == 5:
            return decode_leaf_from_head_logits(*binary_heads)
        shapes = [getattr(arr, "shape", type(arr)) for arr in arrays]
        raise ValueError(f"Unexpected eval logits structure: {shapes}")
    logits = np.asarray(logits)
    return np.argmax(logits, axis=-1)


def flat_labels_from_eval(labels) -> np.ndarray:
    """Trainer may stack multiple label tensors; keep flat leaf ids 0-5."""
    if isinstance(labels, (tuple, list)):
        arrays = [np.asarray(item).reshape(-1) for item in labels]
        return _pick_leaf_label_array(arrays)
    labels = np.asarray(labels)
    if labels.ndim == 2 and labels.shape[0] < labels.shape[1]:
        return _pick_leaf_label_array([np.asarray(row).reshape(-1) for row in labels])
    return labels.reshape(-1)


def _pick_leaf_label_array(arrays: list[np.ndarray]) -> np.ndarray:
    valid = [arr for arr in arrays if arr.size and arr.min() >= 0 and arr.max() <= 5]
    if not valid:
        return arrays[-1].reshape(-1)
    # Leaf labels use the full 0-5 range; binary heads (e.g. r0) only use 0-1.
    return max(valid, key=lambda arr: (int(arr.max()), len(np.unique(arr))))


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = flat_label_predictions_from_logits(logits)
    labels = flat_labels_from_eval(labels)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(
            labels, predictions, labels=FLAT_LABELS, average="macro", zero_division=0
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical multi-head RoBERTa fine-tuning on an H200"
    )
    parser.add_argument("--data-dir", default="pan25-generative-ai-detection-task2-train")
    parser.add_argument("--output-dir", default="roberta_pan_task2_hierarchical_finetune")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tokenizer-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="Resume from a saved checkpoint")
    parser.add_argument("--loss-weight-r0", type=float, default=1.0)
    parser.add_argument("--loss-weight-h0", type=float, default=1.0)
    parser.add_argument("--loss-weight-h2", type=float, default=1.0)
    parser.add_argument("--loss-weight-h3a", type=float, default=1.0)
    parser.add_argument("--loss-weight-h3b", type=float, default=2.0)
    parser.add_argument(
        "--h3b-edited-weight",
        type=float,
        default=8.0,
        help="Cross-entropy weight for H3b class 1 (human-edited, flat label 5)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available: submit this job to an NVIDIA GPU node.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This job requires a GPU that supports BF16 (such as H200).")

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("BF16 supported:", torch.cuda.is_bf16_supported(), flush=True)

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.jsonl"
    dev_path = data_dir / "dev.jsonl"
    for path in (train_path, dev_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset file: {path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_id = "FacebookAI/roberta-base"

    raw = load_dataset(
        "json", data_files={"train": str(train_path), "dev": str(dev_path)}
    )

    def normalize_labels(batch):
        return {"label": [int(value) for value in batch["label"]]}

    raw = raw.map(normalize_labels, batched=True, desc="Normalize labels")
    raw = raw.cast_column("label", ClassLabel(names=[str(i) for i in FLAT_LABELS]))
    train_val = raw["train"].train_test_split(
        test_size=args.val_fraction, seed=args.seed, stratify_by_column="label"
    )
    train_data = train_val["train"]
    val_data = train_val["test"]
    dev_data = raw["dev"]
    print(
        f"Examples: train={len(train_data)}, val={len(val_data)}, dev={len(dev_data)}",
        flush=True,
    )

    def add_head_labels(batch):
        r0, h0, h2, h3a, h3b, leaf = [], [], [], [], [], []
        for label in batch["label"]:
            heads = flat_label_to_heads(label)
            r0.append(heads["r0"])
            h0.append(heads["h0"])
            h2.append(heads["h2"])
            h3a.append(heads["h3a"])
            h3b.append(heads["h3b"])
            leaf.append(heads["leaf_label"])
        return {
            "r0": r0,
            "h0": h0,
            "h2": h2,
            "h3a": h3a,
            "h3b": h3b,
            "leaf_label": leaf,
        }

    train_data = train_data.map(add_head_labels, batched=True, desc="Head labels train")
    val_data = val_data.map(add_head_labels, batched=True, desc="Head labels val")
    dev_data = dev_data.map(add_head_labels, batched=True, desc="Head labels dev")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def tokenize_batch(batch):
        encodings = tokenizer(
            [clean_text(t) for t in batch["text"]],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        for key in ("r0", "h0", "h2", "h3a", "h3b", "leaf_label"):
            encodings[key] = batch[key]
        return encodings

    drop_cols = train_data.column_names
    train_data = train_data.map(
        tokenize_batch,
        batched=True,
        num_proc=args.tokenizer_workers,
        remove_columns=drop_cols,
        desc="Tokenize training set",
    )
    val_data = val_data.map(
        tokenize_batch,
        batched=True,
        num_proc=args.tokenizer_workers,
        remove_columns=val_data.column_names,
        desc="Tokenize validation set",
    )
    dev_data = dev_data.map(
        tokenize_batch,
        batched=True,
        num_proc=args.tokenizer_workers,
        remove_columns=dev_data.column_names,
        desc="Tokenize development set",
    )

    AutoConfig.register("hierarchical_roberta", HierarchicalRobertaConfig)
    config = HierarchicalRobertaConfig.from_pretrained(model_id)
    model = HierarchicalRoberta(config)
    pretrained_roberta = RobertaModel.from_pretrained(model_id)
    model.roberta.load_state_dict(pretrained_roberta.state_dict())
    del pretrained_roberta
    model.head_loss_weights = {
        "r0": args.loss_weight_r0,
        "h0": args.loss_weight_h0,
        "h2": args.loss_weight_h2,
        "h3a": args.loss_weight_h3a,
        "h3b": args.loss_weight_h3b,
    }
    model.register_buffer(
        "h3b_class_weight",
        torch.tensor([1.0, args.h3b_edited_weight], dtype=torch.float32),
        persistent=True,
    )
    print("Trainable parameters:", sum(p.numel() for p in model.parameters()), flush=True)

    steps_per_epoch = max(1, (len(train_data) + args.batch_size - 1) // args.batch_size)
    warmup_steps = max(1, int(0.05 * steps_per_epoch * args.epochs))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        bf16=True,
        fp16=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    collator = HierarchicalDataCollator(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    print("Trainer device:", training_args.device, flush=True)

    train_result = trainer.train(resume_from_checkpoint=True if args.resume else None)
    best_dir = output_dir / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    test_result = trainer.predict(dev_data, metric_key_prefix="dev")
    y_true = flat_labels_from_eval(test_result.label_ids)
    y_pred = flat_label_predictions_from_logits(test_result.predictions)

    report = classification_report(
        y_true,
        y_pred,
        labels=FLAT_LABELS,
        target_names=[str(i) for i in FLAT_LABELS],
        zero_division=0,
        digits=4,
    )
    print("\nPAN development set classification report:\n" + report, flush=True)

    taxonomy_note = (
        "Hierarchical heads: R0 (0 vs rest), H0 (mixed 4), H2 (initiator), "
        "H3a (1 vs 3), H3b (2 vs 5). Inference uses path decoding over six leaves.\n"
    )
    (output_dir / "classification_report.txt").write_text(taxonomy_note + report)
    (output_dir / "confusion_matrix.json").write_text(
        json.dumps(confusion_matrix(y_true, y_pred, labels=FLAT_LABELS).tolist(), indent=2)
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "training": train_result.metrics,
                "development": test_result.metrics,
                "head_loss_weights": model.head_loss_weights,
                "h3b_edited_weight": args.h3b_edited_weight,
            },
            indent=2,
            default=float,
        )
    )
    (output_dir / "training_log.json").write_text(
        json.dumps(trainer.state.log_history, indent=2, default=float)
    )

    with (output_dir / "dev_predictions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["row_index", "true_label", "predicted_label"])
        for i, (actual, predicted) in enumerate(zip(y_true, y_pred)):
            writer.writerow([i, int(actual), int(predicted)])

    print(f"\nFinished. Best model and results saved in: {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
