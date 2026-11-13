"""
train_classifier.py
────────────────────────────────────────────────────────────────────────────────
Fine-tunes google/mobilebert-uncased on backend/data/dataset.csv for 4-class
prompt classification: chat, code, reasoning, document.

Output: backend/core/classifier_model/   (PyTorch checkpoint, ready for ONNX export)

Usage:
    cd <project-root>          # the loqmi/ folder
    python backend/scripts/train_classifier.py

Requirements (install once inside your venv):
    pip install transformers datasets scikit-learn accelerate
────────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
from pathlib import Path

# ── Make sure project root is on sys.path so we can import config ─────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # loqmi/
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_CSV        = PROJECT_ROOT / "backend" / "data" / "dataset.csv"
OUTPUT_DIR      = PROJECT_ROOT / "backend" / "core" / "classifier_model"
BASE_MODEL      = "google/mobilebert-uncased"

# ── Label mapping (must match CLASSIFIER_CATEGORIES in config.py) ─────────────
CATEGORIES = ["chat", "code", "reasoning", "document"]
LABEL2ID   = {label: idx for idx, label in enumerate(CATEGORIES)}
ID2LABEL   = {idx: label for idx, label in enumerate(CATEGORIES)}

# ── Hyperparameters ───────────────────────────────────────────────────────────
MAX_LENGTH      = 128
TRAIN_SPLIT     = 0.85          # 85% train, 15% eval
EPOCHS          = 7
BATCH_SIZE      = 16
LEARNING_RATE   = 2e-5
WEIGHT_DECAY    = 0.01
WARMUP_RATIO    = 0.1
SEED            = 42


# ─────────────────────────────────────────────────────────────────────────────
# 1. Imports
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)

set_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load & validate dataset
# ─────────────────────────────────────────────────────────────────────────────

print("\n[1/6] Loading dataset...")
df = pd.read_csv(DATA_CSV)

# Normalise column names (strip quotes/whitespace just in case)
df.columns = df.columns.str.strip().str.strip('"')

assert "prompt"   in df.columns, "CSV must have a 'prompt' column"
assert "category" in df.columns, "CSV must have a 'category' column"

# Drop rows whose category is not in our 5 classes
before = len(df)
df = df[df["category"].isin(CATEGORIES)].copy()
dropped = before - len(df)
if dropped:
    print(f"  ⚠  Dropped {dropped} rows with unrecognised category labels")

# Drop duplicates & nulls
df = df.dropna(subset=["prompt", "category"])
df = df.drop_duplicates(subset=["prompt"])
df["prompt"]   = df["prompt"].str.strip()
df["label"]    = df["category"].map(LABEL2ID)

print(f"  Total usable rows : {len(df)}")
print("  Class distribution:")
for cat in CATEGORIES:
    n = (df["category"] == cat).sum()
    print(f"    {cat:<12} {n}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Train / eval split
# ─────────────────────────────────────────────────────────────────────────────

print("\n[2/6] Splitting data...")
train_df, eval_df = train_test_split(
    df[["prompt", "label"]],
    test_size=1 - TRAIN_SPLIT,
    random_state=SEED,
    stratify=df["label"],        # keep class ratios equal in both splits
)
print(f"  Train: {len(train_df)}  |  Eval: {len(eval_df)}")

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
eval_dataset  = Dataset.from_pandas(eval_df.reset_index(drop=True))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Tokenise
# ─────────────────────────────────────────────────────────────────────────────

print("\n[3/6] Loading tokeniser & tokenising...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

def tokenize(batch):
    return tokenizer(
        batch["prompt"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,           # DataCollatorWithPadding handles dynamic padding
    )

train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["prompt"])
eval_dataset  = eval_dataset.map(tokenize,  batched=True, remove_columns=["prompt"])

train_dataset = train_dataset.rename_column("label", "labels")
eval_dataset  = eval_dataset.rename_column("label", "labels")

train_dataset.set_format("torch")
eval_dataset.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Model
# ─────────────────────────────────────────────────────────────────────────────

print("\n[4/6] Loading MobileBERT base model...")
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=len(CATEGORIES),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True, 
)
print(f"  Model loaded. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": round(float(acc), 4)}


# ─────────────────────────────────────────────────────────────────────────────
# 7. Training
# ─────────────────────────────────────────────────────────────────────────────

print("\n[5/6] Starting training...")

total_steps     = (len(train_dataset) // BATCH_SIZE) * EPOCHS
warmup_steps    = int(total_steps * WARMUP_RATIO)

training_args = TrainingArguments(
    output_dir                  = str(OUTPUT_DIR),
    num_train_epochs            = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    learning_rate               = LEARNING_RATE,
    weight_decay                = WEIGHT_DECAY,
    warmup_steps                = warmup_steps,
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "accuracy",
    greater_is_better           = True,
    logging_steps               = 20,
    save_total_limit            = 2,            # keep only last 2 checkpoints
    seed                        = SEED,
    report_to                   = "none",       # disable wandb/tensorboard
    fp16                        = False,        # MPS (Apple Silicon) doesn't support fp16
)

trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_dataset,
    eval_dataset    = eval_dataset,
    processing_class       = tokenizer,
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Evaluation — full classification report
# ─────────────────────────────────────────────────────────────────────────────

print("\n[6/6] Running final evaluation...")
predictions = trainer.predict(eval_dataset)
preds  = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids

print("\n" + "─" * 60)
print("CLASSIFICATION REPORT")
print("─" * 60)
print(classification_report(
    labels,
    preds,
    target_names=CATEGORIES,
    digits=3,
))

overall_acc = (preds == labels).mean()
print(f"Overall accuracy : {overall_acc:.1%}")
print("─" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Save best model
# ─────────────────────────────────────────────────────────────────────────────

trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

print(f"\n✅ Model saved to: {OUTPUT_DIR}")
print("   Next step: run export_onnx.py to convert for production inference.\n")