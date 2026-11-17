"""Activation checkpointing for training on CPU, without changing what training computes.

While autograd records, a checkpointed block keeps only its input; its intermediate activations
are recomputed in backward. On CPU the activations live in the pod's RAM: at the delivered batch
sizes they are most of a training's peak and OOM-killed the 2 GiB model pods. This trades them
for ~one more forward pass per step.

The result does not change, bit for bit:
- the recomputation runs the same ops on the same input, with the RNG state of the first pass
  (``torch.utils.checkpoint`` restores it), so dropout / stochastic depth match;
- BatchNorm in train mode would update its running statistics a second time when recomputed:
  they are restored right after the recomputation, so they advance once per step, as without
  checkpointing.

Only ``forward`` is shadowed, on the instances, while the ``with`` block runs: the module tree,
the state_dict keys and everything outside the block (validation, inference) are untouched.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

_NO_FORWARD = object()


def _running_stat_modules(block: nn.Module) -> list[nn.Module]:
    return [
        m for m in block.modules()
        if isinstance(m, nn.modules.batchnorm._BatchNorm)  # pylint: disable=protected-access
        and m.track_running_stats and m.running_mean is not None
    ]


def _fresh(value: Any) -> Any:
    """A new list/tuple/dict holding the same tensors (recursively); anything else as is.

    Each call of a checkpointed block gets its own containers: a block that fills or replaces
    items of its input list in place (pytorchvideo's multipathway blocks do) would otherwise
    hand its own outputs to the recomputation instead of its inputs.
    """
    if isinstance(value, list):
        return [_fresh(v) for v in value]
    if isinstance(value, tuple):
        return type(value)(*(_fresh(v) for v in value)) if hasattr(value, "_fields") \
            else tuple(_fresh(v) for v in value)
    if isinstance(value, dict):
        return {k: _fresh(v) for k, v in value.items()}
    return value


def _checkpointed(block: nn.Module, forward: Callable[..., Any]) -> Callable[..., Any]:
    """``forward`` of ``block`` under checkpoint; a recomputation leaves BatchNorm stats as is."""
    def run(*args: Any, **kwargs: Any) -> Any:
        if not torch.is_grad_enabled():
            return forward(*args, **kwargs)
        calls = [0]

        def once_then_recompute(*inner_args: Any) -> Any:
            calls[0] += 1
            inner_args, inner_kwargs = _fresh(inner_args), _fresh(kwargs)
            if calls[0] == 1:  # the real forward: statistics advance as usual
                return forward(*inner_args, **inner_kwargs)
            norms = _running_stat_modules(block)
            saved = [
                (m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
                for m in norms
            ]
            try:
                return forward(*inner_args, **inner_kwargs)
            finally:  # also on the early stop of a non-reentrant recomputation
                for m, (mean, var, count) in zip(norms, saved):
                    m.running_mean.copy_(mean)
                    m.running_var.copy_(var)
                    m.num_batches_tracked.copy_(count)

        return checkpoint(once_then_recompute, *args, use_reentrant=False)

    return run


@contextmanager
def _shadow_forward(module: nn.Module, forward: Callable[..., Any]) -> Iterator[None]:
    previous = vars(module).get("forward", _NO_FORWARD)
    module.forward = forward
    try:
        yield
    finally:
        if previous is _NO_FORWARD:
            del module.forward
        else:
            module.forward = previous


@contextmanager
def activation_checkpointing(
    blocks: Iterable[nn.Module], *, enabled: bool = True
) -> Iterator[None]:
    """Checkpoint each of ``blocks`` (e.g. a backbone's stages) while the ``with`` block runs.

    Pass ``enabled=False`` (e.g. on CUDA, where activations are not in the pod's RAM) to run
    the block exactly as without it.
    """
    with ExitStack() as stack:
        if enabled:
            for block in blocks:
                stack.enter_context(_shadow_forward(block, _checkpointed(block, block.forward)))
        yield


@contextmanager
def checkpointed_sequential(
    seq: nn.Sequential, ends: Sequence[int], *, enabled: bool = True
) -> Iterator[None]:
    """Run ``seq`` as checkpointed segments ``seq[0:ends[0]], seq[ends[0]:ends[1]], ...``.

    For a flat Sequential (conv, bn, relu, ...) where checkpointing each layer alone would
    save nothing: a segment keeps only its input. The last segment runs to the end of ``seq``.
    """
    if not enabled:
        yield
        return
    bounds = [0, *ends, len(seq)]
    if any(a >= b for a, b in zip(bounds, bounds[1:])):
        raise ValueError(f"Cortes de segmento no válidos para {len(seq)} capas: {list(ends)}")
    segments = [nn.Sequential(*seq[a:b]) for a, b in zip(bounds, bounds[1:])]  # shared layers
    runs = [_checkpointed(segment, segment.forward) for segment in segments]

    def forward(x: torch.Tensor) -> torch.Tensor:
        for run in runs:
            x = run(x)
        return x

    with _shadow_forward(seq, forward):
        yield
