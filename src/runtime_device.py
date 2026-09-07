from __future__ import annotations

import os
from contextlib import nullcontext

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch

def _resolve_device_type() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_local_rank() -> int | None:
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank in (None, ""):
        return None
    try:
        return int(local_rank)
    except ValueError:
        return None


def _null_autocast(*args, **kwargs):
    return nullcontext()


def _resolve_autocast():
    if DEVICE_TYPE == "cuda":
        from torch.amp import autocast as generic_autocast

        def cuda_autocast(*args, **kwargs):
            return generic_autocast("cuda", *args, **kwargs)

        return cuda_autocast
    return _null_autocast


def _resolve_runtime_device() -> torch.device:
    local_rank = _resolve_local_rank()

    if DEVICE_TYPE == "cuda":
        if local_rank is not None:
            torch.cuda.set_device(local_rank)
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cuda")

    return torch.device("cpu")


def move_model_to_runtime(model: torch.nn.Module) -> tuple[torch.nn.Module, str]:
    model = model.to(DEVICE)
    if DEVICE_TYPE != "cpu":
        model = model.to(torch.bfloat16)
    return model, DEVICE_STR


DEVICE_TYPE = _resolve_device_type()
DEVICE = _resolve_runtime_device()
DEVICE_STR = str(DEVICE)
autocast = _resolve_autocast()
