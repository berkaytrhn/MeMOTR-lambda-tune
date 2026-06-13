"""Signal extractors for the B3c signal-driven long-term gate.

Per active track per frame, the SignalGate consumes a 4-vector u_t:
    c_t : matched-detection confidence  (cleanly available from track logits)
    s_t : cosine similarity between f_t (output_embed) and M_{t-1} (long_memory)
    delta_t : IoU(box decoded from M_{t-1}, matched detection)   -- NOT wired
    H_t : normalized entropy of decoder cross-attention for the track query -- NOT wired

delta_t and H_t require plumbing the model does not currently expose
(decoding a box from raw memory; a forward hook on decoder attention). They
are intentionally left unwired here. Per the implementation plan, B3c must
report signal distributions to the human and get go-ahead before training;
attempting to use the unwired signals raises so they are never silently faked.

Normalization (subtract mean, divide std) must use stats computed once on a
held-out train slice and passed in via `mean`/`std`.
"""
import torch

from models.utils import logits_to_scores

SIGNAL_NAMES = ("c_t", "s_t", "delta_t", "H_t")


def confidence_signal(logits: torch.Tensor) -> torch.Tensor:
    """c_t: max class probability per track. Shape (N,)."""
    return torch.max(logits_to_scores(logits=logits), dim=1).values


def cosine_signal(output_embed: torch.Tensor, long_memory: torch.Tensor) -> torch.Tensor:
    """s_t: cosine similarity between f_t and M_{t-1}. Shape (N,)."""
    return torch.nn.functional.cosine_similarity(output_embed, long_memory, dim=-1)


def build_signals(logits: torch.Tensor,
                  output_embed: torch.Tensor,
                  long_memory: torch.Tensor,
                  *,
                  use_delta: bool = False,
                  use_entropy: bool = False,
                  mean: torch.Tensor | None = None,
                  std: torch.Tensor | None = None) -> torch.Tensor:
    """Stack the four signals into u_t of shape (N, 4), optionally normalized.

    use_delta / use_entropy must stay False until delta_t / H_t are wired;
    they raise otherwise so the gate never trains on fabricated signals.
    """
    n = output_embed.shape[0]
    device = output_embed.device

    c_t = confidence_signal(logits)
    s_t = cosine_signal(output_embed, long_memory)

    if use_delta:
        raise NotImplementedError(
            "delta_t (IoU of box decoded from M_{t-1}) is not wired; "
            "decoding a box from raw memory needs the bbox head + decoder. "
            "Report to the human before enabling (see implementation plan B3c).")
    if use_entropy:
        raise NotImplementedError(
            "H_t (decoder cross-attention entropy) is not wired; "
            "it needs a forward hook exposing attention weights. "
            "Report to the human before enabling (see implementation plan B3c).")

    delta_t = torch.zeros(n, device=device)
    H_t = torch.zeros(n, device=device)

    u = torch.stack((c_t, s_t, delta_t, H_t), dim=-1)  # (N, 4)
    if mean is not None and std is not None:
        u = (u - mean) / (std + 1e-6)
    return u
