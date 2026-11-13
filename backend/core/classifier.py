from __future__ import annotations

import logging

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from config import CLASSIFIER_ONNX_DIR, CLASSIFIER_CATEGORIES

logger = logging.getLogger(__name__)

_tokenizer = None
_session = None


def _load() -> None:
    global _tokenizer, _session
    if _session is None:
        logger.info("Loading ONNX classifier...")
        _tokenizer = Tokenizer.from_file(
            str(CLASSIFIER_ONNX_DIR / "tokenizer.json")
        )
        _tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        _tokenizer.enable_truncation(max_length=128)

        model_path = str(CLASSIFIER_ONNX_DIR / "model.onnx")
        _session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        logger.info("Classifier ready")


def classify_with_confidence(prompt: str) -> tuple[str, float]:
    _load()

    encoding = _tokenizer.encode(prompt)

    input_ids      = np.array([encoding.ids],            dtype=np.int64)
    attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
    token_type_ids = np.array([encoding.type_ids],       dtype=np.int64)

    ort_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "token_type_ids" in [inp.name for inp in _session.get_inputs()]:
        ort_inputs["token_type_ids"] = token_type_ids

    logits = _session.run(None, ort_inputs)[0][0]
    exp = np.exp(logits - np.max(logits))          # numerically stable softmax
    probs = exp / exp.sum()
    idx = int(np.argmax(probs))
    return CLASSIFIER_CATEGORIES[idx], float(probs[idx])


def classify(prompt: str) -> str:
    """Kept for callers that only need the label (e.g. main.py)."""
    return classify_with_confidence(prompt)[0]