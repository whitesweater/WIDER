#!/usr/bin/env python3
"""Prepare AQuA-RAT data for CODI training and evaluation.

Source: https://huggingface.co/datasets/deepmind/aqua_rat
"""
from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("WIDER_DATA_DIR", ROOT / "data"))


def _format_options(options: object) -> str:
    lines = []
    for index, option in enumerate(options or []):
        text = str(option).strip()
        if not text:
            continue
        label = chr(ord("A") + index)
        if len(text) >= 2 and text[0].upper() in "ABCDE" and text[1] in {")", ":", "."}:
            label = text[0].upper()
            text = text[2:].strip()
        lines.append(f"{label}: {text}")
    return "\n".join(lines)


def _split_steps(rationale: object) -> list[str]:
    text = str(rationale or "").strip()
    if not text:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def _normalise_split(split_data) -> tuple[list[dict], int]:
    records = []
    skipped = 0
    for ex in split_data:
        stem = str(ex.get("question", "")).strip()
        choices = _format_options(ex.get("options"))
        answer = str(ex.get("correct", "")).strip().upper()
        steps = _split_steps(ex.get("rationale"))

        if not stem or not choices or answer not in "ABCDE" or not steps:
            skipped += 1
            continue

        records.append(
            {
                "question": f"Question: {stem}\nChoices:\n{choices}",
                "cot": "\n".join(steps),
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


def prepare_aqua() -> None:
    from datasets import load_dataset

    dataset = load_dataset("aqua_rat", "raw")
    split_map = {
        "train": DATA_DIR / "aqua_train_clean.json",
        "validation": DATA_DIR / "aqua_val_clean.json",
    }
    for split_name, out_path in split_map.items():
        records, skipped = _normalise_split(dataset[split_name])
        _write_records(records, out_path, skipped)


if __name__ == "__main__":
    prepare_aqua()
