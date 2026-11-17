"""Activation checkpointing saves training memory without changing what training computes."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from app.domain.services.activation_checkpointing import (
    activation_checkpointing,
    checkpointed_sequential,
)


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Dropout(0.3),
    )


class _Net(nn.Module):
    """BatchNorm (running stats), dropout (RNG) and in-place ReLU: what the CNN plugins use."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.stages = nn.ModuleList([_block(3, 8), _block(8, 16), _block(16, 16)])
        self.flat = nn.Sequential(*_block(3, 8), *_block(8, 8))  # a flat Sequential, as ml2's
        self.head = nn.Linear(16 + 8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Stages, then the flat Sequential, pooled and classified together."""
        a = x
        for stage in self.stages:
            a = stage(a)
        b = self.flat(x)
        return self.head(torch.cat([a.mean((2, 3)), b.mean((2, 3))], dim=1))


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(1)
    return torch.randn(6, 3, 16, 16, generator=gen), torch.tensor([0, 1, 2, 0, 1, 2])


def _checkpointing(net: _Net, mode: str):
    if mode == "blocks":
        return activation_checkpointing(net.stages)
    if mode == "segments":
        return checkpointed_sequential(net.flat, [3, 7])
    return activation_checkpointing([], enabled=False)


def _train(mode: str, steps: int = 3) -> tuple[_Net, list[float]]:
    net = _Net().train()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    x, y = _batch()
    losses = []
    for step in range(steps):
        torch.manual_seed(10 + step)
        optimizer.zero_grad()
        with _checkpointing(net, mode):
            loss = nn.functional.cross_entropy(net(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return net, losses


@pytest.mark.parametrize("mode", ["blocks", "segments"])
def test_training_is_bit_identical_with_checkpointing(mode):
    """Same losses, weights and BatchNorm running stats (advanced once per step, not twice)."""
    ref, ref_losses = _train("off")
    got, losses = _train(mode)
    assert losses == ref_losses
    expected = ref.state_dict()
    for key, value in got.state_dict().items():
        assert torch.equal(value, expected[key]), key
    assert int(got.stages[0][1].num_batches_tracked) == 3


def _saved_bytes(mode: str) -> int:
    net, total = _Net().train(), [0]

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        total[0] += tensor.numel() * tensor.element_size()
        return tensor

    x, _ = _batch()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        with _checkpointing(net, mode):
            net(x)
    return total[0]


def test_checkpointed_blocks_keep_far_fewer_activations():
    """Each mode keeps less than plain autograd stores for backward (here ~36% and ~77%).

    "blocks" covers the stages, most of the net; "segments" only the flat Sequential.
    """
    plain = _saved_bytes("off")
    assert _saved_bytes("blocks") < plain * 0.5
    assert _saved_bytes("segments") < plain * 0.85


def test_forward_is_restored_after_the_block():
    """Validation and inference after training run the modules' own forward."""
    net = _Net()
    with activation_checkpointing(net.stages), checkpointed_sequential(net.flat, [3, 7]):
        assert all("forward" in vars(stage) for stage in net.stages)
        assert "forward" in vars(net.flat)
    assert not any("forward" in vars(m) for m in net.modules())


def test_disabled_or_without_grad_runs_plain_forward():
    """enabled=False shadows nothing, and under no_grad a checkpointed block is not recorded."""
    net = _Net().eval()
    x, _ = _batch()
    with activation_checkpointing(net.stages, enabled=False):
        assert not any("forward" in vars(m) for m in net.modules())
    expected = net(x)
    with torch.no_grad(), activation_checkpointing(net.stages):
        assert torch.equal(net(x), expected)


def test_segment_cuts_must_be_increasing_and_inside():
    """A cut at 0, past the end or out of order would drop or repeat layers."""
    net = _Net()
    for ends in ([0, 3], [3, 3], [7, 3], [14]):
        with pytest.raises(ValueError):
            with checkpointed_sequential(net.flat, ends):
                pass


class _InPlaceListBlock(nn.Module):
    """Replaces the item of its input list in place, as pytorchvideo's multipathway blocks do."""

    def __init__(self) -> None:
        super().__init__()
        self.conv, self.norm = nn.Conv2d(4, 4, 3, padding=1), nn.BatchNorm2d(4)

    def forward(self, x: list) -> list:
        """Same-shape output written over the input item."""
        x[0] = torch.relu(self.norm(self.conv(x[0])))
        return x


def test_block_that_rewrites_its_input_list_recomputes_from_its_input():
    """Each call gets its own list: the recomputation does not see the block's own output."""
    def train(checkpointed: bool) -> dict:
        torch.manual_seed(0)
        blocks = nn.ModuleList([_InPlaceListBlock(), _InPlaceListBlock()])
        optimizer = torch.optim.SGD(blocks.parameters(), lr=0.1)
        data = torch.randn(3, 4, 8, 8, generator=torch.Generator().manual_seed(1))
        for _ in range(2):
            optimizer.zero_grad()
            with activation_checkpointing(blocks, enabled=checkpointed):
                x = [data]
                for block in blocks:
                    x = block(x)
            x[0].square().mean().backward()
            optimizer.step()
        return blocks.state_dict()

    expected, got = train(False), train(True)
    for key, value in got.items():
        assert torch.equal(value, expected[key]), key
