"""Stage 4 - chat-format dataset loader for `openai/gpt-oss-20b`.

Each Stage 3 example has shape:
    {"system": str, "user": str, "assistant": str, "meta": {...}}

This module turns it into a HuggingFace `Dataset` of token IDs ready for
SFTTrainer / Trainer, applying the model's *harmony* chat template.

Why a custom loader:
- gpt-oss uses the harmony response format (`<|return|>` etc.). Using
  `tokenizer.apply_chat_template` automatically wires the special tokens
  in the right places, so we never hardcode them.
- We mask the loss on system + user tokens (assistant-only loss), which is
  the standard SFT pattern. This is implemented by zeroing the `labels`
  tensor on prompt tokens (set to -100).
- We also expose a `format_for_inference()` helper so inference uses the
  same chat-template as training (no train/inference skew).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.io_utils import read_jsonl

DEFAULT_REASONING_EFFORT = "low"  # mai rapid; rubrica nu cere CoT lung


def example_to_messages(ex: dict[str, Any]) -> list[dict[str, str]]:
    """Convert a Stage 3 example into a list of chat messages."""
    return [
        {"role": "system", "content": ex["system"]},
        {"role": "user", "content": ex["user"]},
        {"role": "assistant", "content": ex["assistant"]},
    ]


def example_prompt_only(ex: dict[str, Any]) -> list[dict[str, str]]:
    """Return only the system+user part (used for inference)."""
    return [
        {"role": "system", "content": ex["system"]},
        {"role": "user", "content": ex["user"]},
    ]


def _ids_from_template_output(out: Any) -> list[int]:
    """Normalise various return shapes of `apply_chat_template(..., tokenize=True)`:
    - some versions return a flat `list[int]`
    - others return `dict` with `input_ids`
    - some return a list with a single tokenizers.Encoding object
    """
    if isinstance(out, list):
        if out and not isinstance(out[0], int):
            # list[Encoding]
            first = out[0]
            if hasattr(first, "ids"):
                return list(first.ids)
            if hasattr(first, "input_ids"):
                return list(first.input_ids)
        return list(out)
    # `BatchEncoding` is a dict-like; treat it as a dict
    try:
        if "input_ids" in out:
            ids = out["input_ids"]
            if ids and isinstance(ids[0], list):
                return list(ids[0])
            return list(ids)
    except (TypeError, KeyError):
        pass
    raise TypeError(f"unexpected apply_chat_template output type: {type(out)}")


def build_dataset(
    *,
    train_path: Path,
    val_path: Path | None,
    tokenizer,
    max_seq_length: int = 2048,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
):
    """Return (train_ds, val_ds) of HuggingFace `Dataset` with `input_ids`,
    `attention_mask`, `labels` (assistant-only loss masking).
    """
    from datasets import Dataset

    def _encode(ex: dict[str, Any]) -> dict[str, Any]:
        prompt_msgs = example_prompt_only(ex)
        full_msgs = example_to_messages(ex)

        prompt_ids = _ids_from_template_output(
            tokenizer.apply_chat_template(
                prompt_msgs,
                add_generation_prompt=True,
                tokenize=True,
                reasoning_effort=reasoning_effort,
            )
        )
        full_ids = _ids_from_template_output(
            tokenizer.apply_chat_template(
                full_msgs,
                add_generation_prompt=False,
                tokenize=True,
                reasoning_effort=reasoning_effort,
            )
        )

        if len(full_ids) > max_seq_length:
            overflow = len(full_ids) - max_seq_length
            full_ids = full_ids[overflow:]
            prompt_ids = prompt_ids[overflow:] if overflow < len(prompt_ids) else []

        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100  # mask system+user tokens

        attention_mask = [1] * len(full_ids)
        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _load(p: Path) -> Dataset:
        rows = list(read_jsonl(p))
        ds = Dataset.from_list(rows)
        ds = ds.map(_encode, remove_columns=ds.column_names, desc=f"tokenizing {p.name}")
        return ds

    train_ds = _load(train_path)
    val_ds = _load(val_path) if val_path and val_path.exists() else None
    return train_ds, val_ds


def format_for_inference(
    *,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> dict[str, Any]:
    """Apply the chat template to produce input_ids ready for `model.generate`.
    Returns a dict that can be unpacked into `model.generate(**inputs)`.
    """
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort=reasoning_effort,
    )
