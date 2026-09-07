#!/usr/bin/env python3
"""Prepare CommonsenseQA data for CODI training and evaluation.

Preferred source is the zen-E/CommonsenseQA-GPT4omini dataset from
HuggingFace because it includes CoT rationales. If that mirror/download is not
available, fall back only to the public CommonsenseQA choices dataset.

Train split → commonsense_train_clean.json
Validation split → commonsense_val_clean.json
"""
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("WIDER_DATA_DIR", ROOT / "data"))


def _format_choices(choices):
    labels = choices.get("label") or choices.get("labels") or []
    texts = choices.get("text") or choices.get("texts") or []
    return "\n".join(f"{label}: {text}" for label, text in zip(labels, texts))


def _normalise_public_commonsenseqa(split_data):
    examples = []
    for ex in split_data:
        question = str(ex["question"]).strip()
        if "choices" in ex:
            question = f"Question: {question}\nChoices:\n{_format_choices(ex['choices'])}"
        answer = str(ex.get("answerKey", ex.get("answer", ""))).strip()
        examples.append({"question": question, "cot": str(ex.get("cot", "")).strip(), "answer": answer})
    return examples


def _load_source():
    from datasets import load_dataset

    try:
        dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")
        return {
            "train": _normalise_public_commonsenseqa(dataset["train"]),
            "validation": _normalise_public_commonsenseqa(dataset["validation"]),
        }, "zen-E/CommonsenseQA-GPT4omini"
    except Exception as exc:
        print(f"[CommonsenseQA] GPT4omini source unavailable: {exc}")

    for dataset_name in ("tau/commonsense_qa", "commonsense_qa"):
        try:
            dataset = load_dataset(dataset_name)
            return {
                "train": _normalise_public_commonsenseqa(dataset["train"]),
                "validation": _normalise_public_commonsenseqa(dataset["validation"]),
            }, dataset_name
        except Exception as exc:
            print(f"[CommonsenseQA] {dataset_name} source unavailable: {exc}")

    raise FileNotFoundError("No HuggingFace/mirror CommonsenseQA source available")


def prepare_commonsenseqa():
    os.makedirs(DATA_DIR, exist_ok=True)
    dataset, source = _load_source()
    print(f"[CommonsenseQA] Using source: {source}")

    for split_name, out_path in [
        ("train", DATA_DIR / "commonsense_train_clean.json"),
        ("validation", DATA_DIR / "commonsense_val_clean.json"),
    ]:
        split_data = dataset[split_name]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(split_data)} examples to {out_path}")


if __name__ == "__main__":
    prepare_commonsenseqa()
