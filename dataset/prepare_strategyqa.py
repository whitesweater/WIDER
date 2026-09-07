#!/usr/bin/env python3
"""Prepare StrategyQA data for CODI training and evaluation."""
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("WIDER_DATA_DIR", ROOT / "data"))

def prepare_strategyqa():
    from datasets import load_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("zen-E/StrategyQA_CoT_GPT4o")
    train_data = dataset["train"]

    all_examples = []
    for ex in train_data:
        question = ex["question"].strip()
        cot = ex["cot"].strip()
        answer = str(ex["answer"]).strip()
        steps = [s.strip() for s in cot.split("\n") if s.strip()]

        all_examples.append({
            "question": question,
            "cot": cot,
            "steps": steps,
            "answer": answer,
        })

    split_idx = int(len(all_examples) * 0.85)
    train_set = all_examples[:split_idx]
    test_set = all_examples[split_idx:]

    for subset, path in [(train_set, DATA_DIR / "strategyqa_train_clean.json"),
                         (test_set, DATA_DIR / "strategyqa_test_clean.json")]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(subset)} examples to {path}")


if __name__ == "__main__":
    prepare_strategyqa()
