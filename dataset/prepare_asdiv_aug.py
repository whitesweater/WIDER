#!/usr/bin/env python3
"""Prepare ASDiv-Aug data for CODI training and evaluation.

Source: https://huggingface.co/datasets/xuyige/ASDiv-Aug

The HuggingFace dataset already uses GSM8K-style answers:
    <<calculation>>
    ####final_answer
"""
from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("WIDER_DATA_DIR", ROOT / "data"))


def _split_answer(answer: object) -> tuple[str, list[str], str] | None:
    text = str(answer or "").strip()
    if not text:
        return None

    if "####" in text:
        cot_text, final_answer = text.rsplit("####", 1)
    else:
        cot_text, final_answer = text, text

    steps = [line.strip() for line in cot_text.splitlines() if line.strip()]
    final_answer = final_answer.replace(",", "").strip()
    if not steps or not final_answer:
        return None

    return "\n".join(steps), steps, final_answer


def _normalise_split(split_data) -> tuple[list[dict], int]:
    records = []
    skipped = 0
    for ex in split_data:
        question = str(ex.get("question", "")).strip()
        parsed = _split_answer(ex.get("answer"))
        if not question or parsed is None:
            skipped += 1
            continue
        cot, steps, answer = parsed
        records.append(
            {
                "question": question,
                "cot": cot,
                "steps": steps,
                "answer": answer,
            }
        )
    return records, skipped


def _write_records(records: list[dict], out_path: Path, skipped: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(records)} examples to {out_path} (skipped {skipped})")


def prepare_asdiv_aug() -> None:
    from datasets import load_dataset

    dataset = load_dataset("xuyige/ASDiv-Aug")
    split_map = {
        "train": DATA_DIR / "asdiv_aug_train_clean.json",
        "test": DATA_DIR / "asdiv_aug_val_clean.json",
    }
    for split_name, out_path in split_map.items():
        records, skipped = _normalise_split(dataset[split_name])
        _write_records(records, out_path, skipped)


if __name__ == "__main__":
    prepare_asdiv_aug()
