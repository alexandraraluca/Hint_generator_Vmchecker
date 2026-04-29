"""Stage 4 - QLoRA fine-tuning of `openai/gpt-oss-20b` on PA hint data.

Usage:
    python -m src.stage4_finetune.train_qlora --config configs/qlora.yaml
    python -m src.stage4_finetune.train_qlora --config configs/qlora.yaml --dry-run

The `--dry-run` mode loads the config, the tokenizer, builds the dataset,
prints shape info, and exits **without loading the 20B base model**. Use
it to validate the data pipeline on a CPU-only machine.

Real training requires a GPU (T4 ~16 GB minimum with seq_len=2048 + grad
checkpointing). On consumer 24 GB cards (RTX 3090/4090) increase
`max_seq_length` to 4096 if needed.

Outputs:
    models/gpt_oss_20b_pa_hints/
    ├── adapter_config.json     (PEFT)
    ├── adapter_model.safetensors
    ├── tokenizer/              (saved alongside for self-contained loading)
    └── trainer_state.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from src.stage4_finetune.data_loader import build_dataset


def _load_config(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _print_dataset_summary(ds, name: str) -> None:
    if ds is None:
        print(f"  {name}: <none>")
        return
    lengths = [len(x) for x in ds["input_ids"]]
    print(f"  {name}: n={len(ds)}, "
          f"len(min/median/max)={min(lengths)}/"
          f"{sorted(lengths)[len(lengths) // 2]}/"
          f"{max(lengths)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/qlora.yaml"))
    parser.add_argument("--dry-run", action="store_true",
                        help="check data pipeline, do not load base model")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    # ---- tokenizer (small download) ----
    from transformers import AutoTokenizer
    print(f"loading tokenizer from {cfg['model']['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["base_model"],
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- data ----
    train_path = Path(cfg["data"]["train_path"])
    val_path = Path(cfg["data"]["val_path"])
    print(f"building dataset (max_seq={cfg['data']['max_seq_length']})")
    train_ds, val_ds = build_dataset(
        train_path=train_path,
        val_path=val_path,
        tokenizer=tokenizer,
        max_seq_length=cfg["data"]["max_seq_length"],
    )
    print("dataset summary:")
    _print_dataset_summary(train_ds, "train")
    _print_dataset_summary(val_ds, "val")

    if args.dry_run:
        print("\n[dry-run] data pipeline OK; exiting without loading base model.")
        return 0

    # ---- base model + 4-bit quantization ----
    import torch
    from transformers import (
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb_dtype = (
        torch.bfloat16
        if cfg["model"]["bnb_4bit_compute_dtype"] == "bfloat16"
        else torch.float16
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["model"]["use_4bit"],
        bnb_4bit_quant_type=cfg["model"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=bnb_dtype,
        bnb_4bit_use_double_quant=True,
    )
    print(f"loading base model {cfg['model']['base_model']} in 4-bit")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=cfg["model"]["attn_implementation"],
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # ---- LoRA adapter ----
    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias=cfg["lora"]["bias"],
        target_modules=cfg["lora"]["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.3f}%)")

    # ---- training args ----
    out = Path(cfg["train"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        num_train_epochs=cfg["train"]["num_train_epochs"],
        learning_rate=cfg["train"]["learning_rate"],
        lr_scheduler_type=cfg["train"]["lr_scheduler_type"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        weight_decay=cfg["train"]["weight_decay"],
        optim=cfg["train"]["optim"],
        max_grad_norm=cfg["train"]["max_grad_norm"],
        bf16=cfg["train"]["bf16"],
        logging_steps=cfg["train"]["logging_steps"],
        save_steps=cfg["train"]["save_steps"],
        save_total_limit=cfg["train"]["save_total_limit"],
        eval_steps=cfg["train"]["eval_steps"],
        eval_strategy=cfg["train"]["eval_strategy"] if val_ds else "no",
        gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
        dataloader_num_workers=cfg["train"]["dataloader_num_workers"],
        seed=cfg["train"]["seed"],
        report_to=cfg["train"]["report_to"],
    )

    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print("starting training...")
    trainer.train()

    print("saving adapter + tokenizer")
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out / "tokenizer"))

    # write a small manifest so inference knows what's what
    manifest = {
        "base_model": cfg["model"]["base_model"],
        "adapter_dir": str(out),
        "max_seq_length": cfg["data"]["max_seq_length"],
        "trained_on": {
            "train": str(train_path),
            "val": str(val_path),
        },
    }
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"done. adapter at {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
