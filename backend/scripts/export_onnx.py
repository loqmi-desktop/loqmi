"""
export_onnx.py
────────────────────────────────────────────────────────────────────────────────
Exports the trained MobileBERT classifier to ONNX format and validates it.
Uses torch.onnx.export directly — bypasses Optimum optimisation which does
not support MobileBERT.

  Source : backend/core/classifier_model/   (PyTorch .safetensors)
  Target : backend/core/classifier_onnx/  

Usage:
    cd backend/scripts
    python export_onnx.py

Requirements:
    pip install transformers optimum[onnxruntime] onnx onnxruntime
────────────────────────────────────────────────────────────────────────────────
"""

import sys
import shutil
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR    = PROJECT_ROOT / "backend" / "core" / "classifier_model"
ONNX_DIR     = PROJECT_ROOT / "backend" / "core" / "classifier_onnx"
ONNX_PATH    = ONNX_DIR / "model.onnx"
ONNX_DIR.mkdir(parents=True, exist_ok=True)

# ── Categories — must match config.py CLASSIFIER_CATEGORIES ──────────────────
CATEGORIES = ["chat", "code", "reasoning", "document"]

# ── Sanity-check prompts (one per category) ───────────────────────────────────
SANITY_PROMPTS = [
    ("What are some tips for being more productive?",                "chat"),
    ("Write a Python function to reverse a linked list.",            "code"),
    ("If a train travels 60 mph for 2.5 hours how far does it go?", "reasoning"),
    ("Summarise the key findings from this report.",                 "document")
]

OPSET = 14   # safe for MobileBERT, well-supported by onnxruntime

# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ─────────────────────────────────────────────────────────────────────────────
# 1. Verify source model
# ─────────────────────────────────────────────────────────────────────────────

print("\n[1/5] Verifying source model...")

required  = ["config.json", "tokenizer.json", "tokenizer_config.json"]
missing   = [f for f in required if not (MODEL_DIR / f).exists()]
has_weights = (MODEL_DIR / "model.safetensors").exists() or \
              (MODEL_DIR / "pytorch_model.bin").exists()

if missing or not has_weights:
    print(f"  ✗  Missing files in {MODEL_DIR}")
    for f in missing:
        print(f"       - {f}")
    if not has_weights:
        print("       - model.safetensors")
    print("\n  Run train_classifier.py first.")
    sys.exit(1)

print(f"  ✓  Source model found at: {MODEL_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load PyTorch model + tokeniser
# ─────────────────────────────────────────────────────────────────────────────

print("\n[2/5] Loading PyTorch model...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
pt_model  = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
pt_model.eval()

id2label = pt_model.config.id2label
assert len(id2label) == 4, f"Expected 4 labels, got {len(id2label)}"
assert set(id2label.values()) == set(CATEGORIES), \
    f"Label mismatch: {set(id2label.values())} vs {set(CATEGORIES)}"

print(f"  ✓  Label mapping: {id2label}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Export to ONNX via torch.onnx.export
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[3/5] Exporting to ONNX (opset {OPSET})...")

# Back up old model
if ONNX_PATH.exists():
    backup = ONNX_DIR / "model.onnx.bak"
    shutil.copy2(ONNX_PATH, backup)
    print(f"  ⚠  Backed up old model.onnx → model.onnx.bak")

# Build a dummy input for tracing
dummy = tokenizer(
    "classify this prompt",
    return_tensors="pt",
    truncation=True,
    max_length=128,
    padding="max_length",
)

input_names  = ["input_ids", "attention_mask", "token_type_ids"]
output_names = ["logits"]
dynamic_axes = {
    "input_ids":      {0: "batch", 1: "seq"},
    "attention_mask": {0: "batch", 1: "seq"},
    "token_type_ids": {0: "batch", 1: "seq"},
    "logits":         {0: "batch"},
}

with torch.no_grad():
    torch.onnx.export(
        pt_model,
        args=(dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
        f=str(ONNX_PATH),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        do_constant_folding=True,
    )

size_mb = ONNX_PATH.stat().st_size / (1024 * 1024)
print(f"  ✓  ONNX model written: {ONNX_PATH}  ({size_mb:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Validate ONNX model with onnxruntime directly
#    (skip ORTModelForSequenceClassification — use raw onnxruntime session
#     so we don't need Optimum to support MobileBERT)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[4/5] Sanity check — PyTorch vs ONNX predictions...")

import onnxruntime as ort

session = ort.InferenceSession(
    str(ONNX_PATH),
    providers=["CPUExecutionProvider"],
)

print(f"\n  {'Prompt':<52} {'Expected':<12} {'PyTorch':<12} {'ONNX':<12} {'Match'}")
print(f"  {'─'*52} {'─'*12} {'─'*12} {'─'*12} {'─'*5}")

all_pass = True

for prompt, expected in SANITY_PROMPTS:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length",
    )

    # PyTorch
    with torch.no_grad():
        pt_logits = pt_model(**enc).logits
    pt_pred = CATEGORIES[pt_logits.argmax(-1).item()]

    # ONNX via raw onnxruntime
    ort_inputs = {
        "input_ids":      enc["input_ids"].numpy(),
        "attention_mask": enc["attention_mask"].numpy(),
        "token_type_ids": enc["token_type_ids"].numpy(),
    }
    ort_logits = session.run(["logits"], ort_inputs)[0]
    onnx_pred  = CATEGORIES[int(np.argmax(ort_logits, axis=-1)[0])]

    match = "✓" if pt_pred == onnx_pred else "✗"
    if pt_pred != onnx_pred:
        all_pass = False

    short = (prompt[:49] + "...") if len(prompt) > 52 else prompt
    print(f"  {short:<52} {expected:<12} {pt_pred:<12} {onnx_pred:<12} {match}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Copy tokeniser files
# ─────────────────────────────────────────────────────────────────────────────

print("\n[5/5] Verifying tokeniser files...")

tokenizer_files = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
    "config.json",
]

for fname in tokenizer_files:
    src  = MODEL_DIR / fname
    dest = ONNX_DIR  / fname
    if not dest.exists() and src.exists():
        shutil.copy2(src, dest)
        print(f"  Copied  {fname}")
    elif dest.exists():
        print(f"  ✓  {fname}")
    else:
        print(f"  ⚠  {fname} not found in either location")

print(f"\n  Files in classifier_onnx/:")
for f in sorted(ONNX_DIR.iterdir()):
    if f.name == "model.onnx.bak":
        continue
    size_kb = f.stat().st_size // 1024
    print(f"    {f.name:<35} {size_kb:>6} KB")


# ─────────────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
if all_pass:
    print("✅ Export complete. Production classifier is ready.")
    print(f"   ONNX model : {ONNX_PATH}")
    print("   No changes needed in classifier.py or router.py.")
    print("   Restart your server — new 4-class model is live.\n")
else:
    print("⚠  Export finished but sanity check had mismatches.")
    print("   Do not deploy until mismatches are investigated.\n")