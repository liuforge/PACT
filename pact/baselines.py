"""TimeCMA (Liu et al., AAAI 2025) re-implemented on top of the shared data/embedding pipeline.

Ported from https://github.com/ChenxiLiu-HNU/TimeCMA (models/TimeCMA.py, layers/Cross_Modal_Align.py),
Copyright 2025 S-Lab, S-Lab License 1.0 (non-commercial use). The single-layer, single-head pre-norm
CrossModal block is written with nn.MultiheadAttention, which is mathematically identical for
n_layers=1 (residual attention scores are only carried between layers).
Embeddings are the same last-token prompt embeddings PACT uses, so both models see identical inputs.
"""
import torch.nn as nn

from pact.model import RevIN


class CrossModal(nn.Module):
    """Pre-norm Transformer block whose attention takes (q, k, v) from different modalities."""

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, 1, dropout=dropout, batch_first=True)
        self.drop_attn = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.drop_ffn = nn.Dropout(dropout)

    def forward(self, q, kv):
        q, kv = self.norm_attn(q), self.norm_attn(kv)
        q = q + self.drop_attn(self.attn(q, kv, kv, need_weights=False)[0])
        return q + self.drop_ffn(self.ff(self.norm_ffn(q)))


class TimeCMA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        N, L, H, C, E = cfg.num_vars, cfg.seq_len, cfg.pred_len, cfg.d_model, cfg.d_llm
        self.revin = RevIN(N, affine=False)
        self.length_to_feature = nn.Linear(L, C)
        enc = nn.TransformerEncoderLayer(C, cfg.n_heads, dropout=cfg.dropout, batch_first=True, norm_first=True)
        self.ts_encoder = nn.TransformerEncoder(enc, cfg.e_layers, enable_nested_tensor=False)
        penc = nn.TransformerEncoderLayer(E, cfg.n_heads, dropout=cfg.dropout, batch_first=True, norm_first=True)
        self.prompt_encoder = nn.TransformerEncoder(penc, cfg.e_layers, enable_nested_tensor=False)
        self.cross = CrossModal(N, cfg.tc_d_ff, cfg.dropout)
        dec = nn.TransformerDecoderLayer(C, cfg.n_heads, dropout=cfg.dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, cfg.d_layers)
        self.c_to_length = nn.Linear(C, H)

    def forward(self, x, emb=None):  # x [B, L, N], emb [B, N, E]
        z = self.revin.norm(x).permute(0, 2, 1)  # [B, N, L]
        enc = self.ts_encoder(self.length_to_feature(z)).permute(0, 2, 1)  # [B, C, N]
        prompt = self.prompt_encoder(emb).permute(0, 2, 1)  # [B, E, N]
        cross = self.cross(enc, prompt).permute(0, 2, 1)  # [B, N, C]
        dec = self.decoder(cross, cross)
        y = self.c_to_length(dec).permute(0, 2, 1)  # [B, H, N]
        return self.revin.denorm(y), {}
