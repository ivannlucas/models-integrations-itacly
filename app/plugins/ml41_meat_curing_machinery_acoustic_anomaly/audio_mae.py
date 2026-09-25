"""Audio-MAE (Masked Autoencoder with ViT backbone) architecture.

Ported verbatim from inbox/a41/codigo/src/audio_mae.py (original comment: "Based on
official Audio-MAE / MAE (He et al., 2022) implementation"). Do not modify — the
delivered checkpoints were trained against this exact module graph, and load_state_dict
will fail on any shape-affecting change.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """(freq, time) Image to Patch Embedding."""

    def __init__(self, img_size=(128, 1024), patch_size=(16, 16), in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, E, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    """Generate 2D sinusoidal positional embeddings.

    Returns: (grid_size_h * grid_size_w, embed_dim) or (1 + grid_size_h * grid_size_w, embed_dim)
    """
    grid_h = torch.arange(grid_size_h, dtype=torch.float32)
    grid_w = torch.arange(grid_size_w, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")  # (H, W) each
    grid = torch.stack(grid, dim=0).reshape(2, -1)  # (2, H*W)

    embed_dim_half = embed_dim // 2
    omega = torch.arange(embed_dim_half // 2, dtype=torch.float32) / (embed_dim_half // 2)
    omega = 1.0 / (10000 ** omega)  # (D/4,)

    out_h = grid[0:1].T @ omega.unsqueeze(0)  # (H*W, D/4)
    out_w = grid[1:2].T @ omega.unsqueeze(0)  # (H*W, D/4)

    pos_embed = torch.cat([
        torch.sin(out_h), torch.cos(out_h),
        torch.sin(out_w), torch.cos(out_w),
    ], dim=1)  # (H*W, D)

    if cls_token:
        pos_embed = torch.cat([torch.zeros(1, embed_dim), pos_embed], dim=0)

    return pos_embed


class AudioMAE(nn.Module):
    """Masked Autoencoder (MAE) with Transformer backbone, adapted for audio spectrograms."""

    def __init__(self, img_size=(128, 1024), patch_size=(16, 16), in_chans=1,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=True):
        super().__init__()

        self.norm_pix_loss = norm_pix_loss

        # --- ENCODER ---
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches  # noqa: F841 (kept for parity with original)
        grid_h, grid_w = self.patch_embed.grid_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w, cls_token=True)
        self.pos_embed = nn.Parameter(pos_embed.unsqueeze(0), requires_grad=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation="gelu", batch_first=True, norm_first=True,
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = norm_layer(embed_dim)

        # --- DECODER ---
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        decoder_pos_embed = get_2d_sincos_pos_embed(decoder_embed_dim, grid_h, grid_w, cls_token=True)
        self.decoder_pos_embed = nn.Parameter(decoder_pos_embed.unsqueeze(0), requires_grad=False)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim, nhead=decoder_num_heads,
            dim_feedforward=int(decoder_embed_dim * mlp_ratio),
            activation="gelu", batch_first=True, norm_first=True,
            dropout=0.0,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0] * patch_size[1] * in_chans, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.view(m.weight.size(0), -1))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def random_masking(self, x, mask_ratio):
        """Per-sample random masking by per-sample shuffling. x: [N, L, D]."""
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)  # (N, L, E)
        x = x + self.pos_embed[:, 1:, :]

        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.encoder(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed

        x = self.decoder(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """imgs: [N, 1, H, W]; pred: [N, L, p*p*1]; mask: [N, L], 0=keep, 1=remove."""
        target = self.patchify(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on masked patches only
        return loss

    def patchify(self, imgs):
        """imgs: (N, 1, H, W) -> x: (N, L, patch_size_h * patch_size_w * 1)."""
        p_h, p_w = self.patch_embed.patch_size
        h = self.patch_embed.img_size[0] // p_h
        w = self.patch_embed.img_size[1] // p_w
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p_h, w, p_w))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p_h * p_w * 1))
        return x

    def unpatchify(self, x):
        """x: (N, L, patch_size_h * patch_size_w * 1) -> imgs: (N, 1, H, W)."""
        p_h, p_w = self.patch_embed.patch_size
        h = self.patch_embed.img_size[0] // p_h
        w = self.patch_embed.img_size[1] // p_w
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p_h, p_w, 1))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p_h, w * p_w))
        return imgs

    def forward(self, x, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask
