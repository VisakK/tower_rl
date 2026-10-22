"""Keypoint network: a ResNet-34 U-Net (ImageNet-pretrained torchvision encoder) giving full-resolution logits for
the block corners, the block edges, the visible-block mask and the arm-hidden-block mask of one camera image, plus
their sub-pixel decoding.

Heatmap targets are Gaussians (corners sigma 1.5 px, edges 1.0 px) around the projected visible corners / edge
points, so a log-parabola through the three samples around a maximum recovers its sub-pixel position exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from tower_rl.perception.data import IMAGENET_MEAN, IMAGENET_STD

HEADS = ("corner", "edge", "mask", "hidden")  # hidden: a block surface covered by an arm or finger (ids 254)
SIGMA = {"corner": 1.5, "edge": 1.0}


def conv_bn(i: int, o: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Up(nn.Module):
    def __init__(self, i: int, skip: int, o: int):
        super().__init__()
        self.conv = nn.Sequential(conv_bn(i + skip, o), conv_bn(o, o))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class KeypointNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        r = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)  # /2, 64
        self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4  # /4 64, /8 128, /16 256, /32 512
        self.full = nn.Sequential(conv_bn(3, 24), conv_bn(24, 24))  # full-resolution detail for sub-pixel heads
        self.u4, self.u3, self.u2 = Up(512, 256, 256), Up(256, 128, 128), Up(128, 64, 96)
        self.u1, self.u0 = Up(96, 64, 64), Up(64, 24, 32)
        self.head = nn.Sequential(conv_bn(32, 32), nn.Conv2d(32, len(HEADS), 1))
        nn.init.constant_(self.head[-1].bias, -4.0)  # rare positives: start near p = 0.02
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x [N, 3, H, W]`` RGB in [0, 1] -> logits ``[N, 4, H, W]`` (``HEADS``)."""
        x = (x - self.mean) / self.std
        f0 = self.full(x)
        s1 = self.stem(x)
        s2 = self.l1(self.pool(s1))
        s3 = self.l2(s2)
        s4 = self.l3(s3)
        y = self.u4(self.l4(s4), s4)
        y = self.u1(self.u2(self.u3(y, s3), s2), s1)
        return self.head(self.u0(y, f0))


def quality_focal(logits: torch.Tensor, target: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
    """Focal loss with soft (Gaussian) targets, normalized by the number of near-peak pixels."""
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    w = (logits.sigmoid() - target).abs().pow(beta)
    return (ce * w).sum() / (target > 0.5).sum().clamp_min(1.0)


@torch.no_grad()
def peaks(prob: torch.Tensor, thr: float = 0.3, nms: int = 5, max_n: int = 400) -> torch.Tensor:
    """Sub-pixel local maxima ``[n, 3]`` (x, y, score) of one heatmap ``[H, W]`` (probabilities)."""
    h, w = prob.shape
    m = F.max_pool2d(prob[None, None], nms, 1, nms // 2)[0, 0]
    keep = (prob >= m) & (prob > thr)
    keep[0, :] = keep[-1, :] = keep[:, 0] = keep[:, -1] = False
    ys, xs = torch.nonzero(keep, as_tuple=True)
    if len(ys) > max_n:
        top = prob[ys, xs].topk(max_n).indices
        ys, xs = ys[top], xs[top]
    lp = prob.clamp_min(1e-6).log()

    def vertex(a, b, c):
        den = a - 2 * b + c
        return torch.where(den < -1e-6, 0.5 * (a - c) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)

    dx = vertex(lp[ys, xs - 1], lp[ys, xs], lp[ys, xs + 1])
    dy = vertex(lp[ys - 1, xs], lp[ys, xs], lp[ys + 1, xs])
    return torch.stack([xs.float() + dx, ys.float() + dy, prob[ys, xs]], -1)
