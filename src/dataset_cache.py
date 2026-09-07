from __future__ import annotations

import hashlib
import os
import uuid
import warnings
from pathlib import Path
from typing import Any, Mapping


DATASET_CACHE_VERSION = 2


def resolve_dataset_cache_dir(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    if cache_dir:
        return Path(cache_dir)

    explicit_dir = os.environ.get("WIDER_DATASET_CACHE_DIR")
    if explicit_dir:
        return Path(explicit_dir)

    cache_root = os.environ.get("WIDER_CACHE_DIR")
    if cache_root:
        return Path(cache_root) / "dataset_cache"

    return Path(__file__).resolve().parents[1] / "cache" / "dataset_cache"


def _cache_key_parts(
    *,
    data_name: str,
    raw_data_path: str | os.PathLike[str],
    tokenizer_name: str,
    tokenizer_vocab_size: int,
    tokenizer_pad_token_id: int | None,
    tokenizer_bos_token_id: int | None,
    tokenizer_eos_token_id: int | None,
    bot_id: int,
    eot_id: int,
    settings: Mapping[str, Any],
) -> list[str]:
    raw_path = Path(raw_data_path).resolve()
    stat = raw_path.stat()
    parts = [
        f"version={DATASET_CACHE_VERSION}",
        f"data_name={data_name}",
        f"raw_path={raw_path}",
        f"raw_size={stat.st_size}",
        f"raw_mtime_ns={stat.st_mtime_ns}",
        f"tokenizer_name={tokenizer_name}",
        f"tokenizer_vocab_size={tokenizer_vocab_size}",
        f"tokenizer_pad_token_id={tokenizer_pad_token_id}",
        f"tokenizer_bos_token_id={tokenizer_bos_token_id}",
        f"tokenizer_eos_token_id={tokenizer_eos_token_id}",
        f"bot_id={bot_id}",
        f"eot_id={eot_id}",
    ]
    for key in sorted(settings):
        parts.append(f"{key}={settings[key]}")
    return parts


def build_dataset_cache_path(
    *,
    data_name: str,
    raw_data_path: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str] | None,
    tokenizer_name: str,
    tokenizer_vocab_size: int,
    tokenizer_pad_token_id: int | None,
    tokenizer_bos_token_id: int | None,
    tokenizer_eos_token_id: int | None,
    bot_id: int,
    eot_id: int,
    settings: Mapping[str, Any],
) -> Path:
    digest = hashlib.sha256(
        "\n".join(
            _cache_key_parts(
                data_name=data_name,
                raw_data_path=raw_data_path,
                tokenizer_name=tokenizer_name,
                tokenizer_vocab_size=tokenizer_vocab_size,
                tokenizer_pad_token_id=tokenizer_pad_token_id,
                tokenizer_bos_token_id=tokenizer_bos_token_id,
                tokenizer_eos_token_id=tokenizer_eos_token_id,
                bot_id=bot_id,
                eot_id=eot_id,
                settings=settings,
            )
        ).encode("utf-8")
    ).hexdigest()[:16]
    return resolve_dataset_cache_dir(cache_dir) / f"{data_name}_{digest}.pt"


def _torch_load_fast(cache_path: Path):
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    def _torch_load_plain():
        try:
            return torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(cache_path, map_location="cpu")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Detected pickle protocol .* weights_only Unpickler",
            category=UserWarning,
        )
        try:
            return torch.load(cache_path, map_location="cpu", mmap=True, weights_only=True)
        except TypeError:
            try:
                return torch.load(cache_path, map_location="cpu", weights_only=True)
            except TypeError:
                return _torch_load_plain()
        except Exception:
            try:
                return torch.load(cache_path, map_location="cpu", weights_only=True)
            except Exception:
                return _torch_load_plain()


def load_dataset_cache(cache_path: str | os.PathLike[str]) -> dict[str, Any] | None:
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        return None

    try:
        cached = _torch_load_fast(cache_path)
    except Exception:
        return None

    if not isinstance(cached, dict):
        return None
    if "data_dict" not in cached or "keys" not in cached:
        return None
    return cached


def should_save_dataset_cache() -> bool:
    rank = os.environ.get("RANK")
    if rank not in (None, ""):
        return rank == "0"

    local_rank = os.environ.get("LOCAL_RANK")
    return local_rank in (None, "", "0")


def save_dataset_cache(cache_path: str | os.PathLike[str], data_dict: Mapping[str, Any], keys: list[str]) -> None:
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(
            {"data_dict": data_dict, "keys": list(keys)},
            tmp_path,
            pickle_protocol=4,
        )
        tmp_path.replace(cache_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
