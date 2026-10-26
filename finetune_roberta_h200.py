#!/usr/bin/env python3
"""Fully fine-tune RoBERTa-base on PAN 2025 Task 2 using one NVIDIA H200."""

import argparse
import csv
import json
import os
import re
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from datasets import ClassLabel, load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def clean_text(text):
    text = str(text).replace("_SEP_", " ").replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", text).strip()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(
            labels, predictions, labels=list(range(6)), average="macro", zero_division=0
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Full RoBERTa fine-tuning on an H200")
    parser.add_argument("--data-dir", default="pan25-generative-ai-detection-task2-train")
    parser.add_argument("--output-dir", default="roberta_pan_task2_full_finetune")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tokenizer-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="Resume from a saved checkpoint")
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
    labels = list(range(6))

    # Read ALL examples, not a debug subset. Keep the provided development
    # split separate so it is only used for final evaluation.
    raw = load_dataset(
        "json", data_files={"train": str(train_path), "dev": str(dev_path)}
    )

    def normalize_labels(batch):
        return {"label": [int(value) for value in batch["label"]]}

    raw = raw.map(normalize_labels, batched=True, desc="Normalize labels")
    raw = raw.cast_column("label", ClassLabel(names=[str(i) for i in labels]))
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

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def tokenize_batch(batch):
        encodings = tokenizer(
            [clean_text(t) for t in batch["text"]],
            truncation=True,
            max_length=args.max_length,
            padding=False,  # Dynamic padding happens in the data collator.
        )
        encodings["labels"] = batch["label"]
        return encodings

    # Hugging Face datasets automatically caches map() results for later runs.
    train_data = train_data.map(
        tokenize_batch,
        batched=True,
        num_proc=args.tokenizer_workers,
        remove_columns=train_data.column_names,
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

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(labels),
        id2label={i: str(i) for i in labels},
        label2id={str(i): i for i in labels},
    )
    # Full fine-tuning: DO NOT freeze model.roberta or other parameters.
    print("Trainable parameters:", sum(p.numel() for p in model.parameters()), flush=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=0.05,
        lr_scheduler_type="linear",
        bf16=True,                       # H200 supports BF16 mixed precision.
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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    print("Trainer device:", training_args.device, flush=True)

    # Resuming restores model, optimizer, LR schedule, and progress where possible.
    train_result = trainer.train(resume_from_checkpoint=True if args.resume else None)
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))

    # Final test evaluation on the provided PAN development dataset.
    test_result = trainer.predict(dev_data, metric_key_prefix="dev")
    y_true = test_result.label_ids
    y_pred = np.argmax(test_result.predictions, axis=-1)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[str(i) for i in labels],
        zero_division=0,
        digits=4,
    )
    print("\nPAN development set classification report:\n" + report, flush=True)

    # Save reusable metrics, confusion matrix, logs and individual predictions.
    (output_dir / "classification_report.txt").write_text(report)
    (output_dir / "confusion_matrix.json").write_text(
        json.dumps(confusion_matrix(y_true, y_pred, labels=labels).tolist(), indent=2)
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {"training": train_result.metrics, "development": test_result.metrics},
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
