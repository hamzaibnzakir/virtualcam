"""The small swap model ("student"). Same inputs and output as inswapper_128, so the VirtualCam app can
load it in its place:
  target: (N,3,128,128) RGB in 0..1, the face crop aligned the insightface way
  source: (N,512) the identity latent (ArcFace embedding x emap, normalised)
  output: (N,3,128,128) RGB in 0..1
Same layout as the teacher (encoder -> identity-modulated residual blocks at 32x32 -> decoder), much
narrower, and it goes down to low resolution sooner, where the teacher spends most of its time.
"""
import torch
from torch import nn
import torch.nn.functional as F


class StyleConv(nn.Module):
    """Reflect-padded 3x3 conv, instance norm, then scale/shift from the identity (as the teacher)."""

    def __init__(self, ch, latent=512):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1, padding_mode='reflect')
        self.style = nn.Linear(latent, ch * 2)

    def forward(self, x, z):
        x = self.conv(x)
        x = F.instance_norm(x)
        scale, shift = self.style(z).chunk(2, dim=1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class StyleBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.a = StyleConv(ch)
        self.b = StyleConv(ch)

    def forward(self, x, z):
        return x + self.b(F.relu(self.a(x, z)), z)


class Student(nn.Module):
    def __init__(self, width=256, blocks=6, base=32):
        super().__init__()
        act = lambda: nn.LeakyReLU(0.2)  # noqa: E731
        self.enc = nn.Sequential(
            nn.Conv2d(3, base, 7, padding=3, padding_mode='reflect'), act(),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1), act(),        # 64x64
            nn.Conv2d(base * 2, width, 3, stride=2, padding=1), act(),      # 32x32
        )
        self.blocks = nn.ModuleList(StyleBlock(width) for _ in range(blocks))
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(width, base * 4, 3, padding=1), act(),                # 64x64
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base * 4, base * 2, 3, padding=1), act(),             # 128x128
            nn.Conv2d(base * 2, base, 3, padding=1), act(),
            nn.Conv2d(base, 3, 7, padding=3, padding_mode='reflect'),
        )

    def forward(self, target, source):
        x = self.enc(target * 2 - 1)
        for block in self.blocks:
            x = block(x, source)
        # Predict the change from the input face: most of the picture (light, pose, background) stays.
        return (target + self.dec(x)).clamp(0, 1)


def gflops(model, size=128):
    """Multiply-adds x2 for convs and linears at batch 1."""
    total = 0
    hooks = []

    def conv_hook(m, i, o):
        nonlocal total
        total += 2 * o.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def lin_hook(m, i, o):
        nonlocal total
        total += 2 * m.in_features * m.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    with torch.no_grad():
        model(torch.rand(1, 3, size, size), torch.randn(1, 512))
    for h in hooks:
        h.remove()
    return total / 1e9


if __name__ == '__main__':
    for w in (192, 256, 384):
        m = Student(width=w).eval()
        print(w, f'{gflops(m):.1f} GFLOPs', f'{sum(p.numel() for p in m.parameters()) / 1e6:.1f} M params')
