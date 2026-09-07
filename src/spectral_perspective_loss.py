import torch
import torch.nn.functional as F


def spectral_perspective_loss(Z: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    ReSN-style spectral regularization adapted for latent token matrices.

    The column-sum of the row-normalized latent matrix approximates its
    dominant right singular vector. We project every normalized row onto that
    direction and penalise the squared magnitude. Row normalization keeps the
    regularizer focused on spectral shape rather than absolute hidden-state
    norm, avoiding domination of the main training objective.

    Z: (n_tokens, hidden_dim)
    """
    normalized_Z = F.normalize(Z.float(), p=2, dim=-1, eps=eps)
    col_sum = normalized_Z.sum(dim=0, keepdim=True).t()  # (D, 1)
    q = col_sum / (torch.sqrt((col_sum ** 2).sum()) + eps)
    q = q.detach()                                     # stop gradient
    regterm = normalized_Z @ q                         # (N, 1)
    return (regterm ** 2).mean().to(Z.dtype)
