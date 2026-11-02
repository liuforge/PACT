"""TimeCMA (Liu et al., AAAI 2025) re-implemented on top of the shared data/embedding pipeline.

Ported from https://github.com/ChenxiLiu-HNU/TimeCMA (models/TimeCMA.py, layers/Cross_Modal_Align.py),
Copyright 2025 S-Lab, S-Lab License 1.0 (non-commercial use). The single-layer, single-head pre-norm
CrossModal block is written with nn.MultiheadAttention, which is mathematically identical for
n_layers=1 (residual attention scores are only carried between layers).
Embeddings are the same last-token prompt embeddings PACT uses, so both models see identical inputs.
"""
import torch
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

    def forward(self, x, emb=None, x_mark=None):  # x [B, L, N], emb [B, N, E]
        z = self.revin.norm(x).permute(0, 2, 1)  # [B, N, L]
        enc = self.ts_encoder(self.length_to_feature(z)).permute(0, 2, 1)  # [B, C, N]
        prompt = self.prompt_encoder(emb).permute(0, 2, 1)  # [B, E, N]
        cross = self.cross(enc, prompt).permute(0, 2, 1)  # [B, N, C]
        dec = self.decoder(cross, cross)
        y = self.c_to_length(dec).permute(0, 2, 1)  # [B, H, N]
        return self.revin.denorm(y), {}


class V5(nn.Module):
    """The model used for the first submission (TimeCMA_pro_v5.Dual_Decomposed_Advanced), unchanged.

    Kept to report how the original architecture behaves under the corrected protocol. It contains the
    LocalDetailRefinement CNN branch and the training-only decomposition constraints of the original code;
    its auxiliary loss (0.1 * InfoNCE + 0.05 * orthogonality, fixed inside the model) is added with weight 1.
    """

    def __init__(self, cfg):
        super().__init__()
        from pact.legacy.v5_model import Dual_Decomposed_Advanced
        self.net = Dual_Decomposed_Advanced(
            device=cfg.device_str, channel=cfg.d_model, num_nodes=cfg.num_vars, seq_len=cfg.seq_len,
            pred_len=cfg.pred_len, dropout_n=cfg.dropout, d_llm=cfg.d_llm, e_layer=cfg.e_layers,
            d_layer=cfg.d_layers, head=cfg.n_heads, kernel_size=cfg.kernel_size, learnable_decomposition=True,
            fusion_strategy=cfg.v5_fusion, ts_encoder_type=cfg.v5_encoder, n_prototypes=cfg.n_prototypes)

    def forward(self, x, emb=None, x_mark=None):
        pred, aux = self.net(x, None, emb)
        return pred, ({"legacy": aux} if torch.is_tensor(aux) else {})


class TSLib(nn.Module):
    """PatchTST / iTransformer / DLinear / TimesNet from Time-Series-Library (MIT, see pact/tslib/)."""

    def __init__(self, cfg):
        super().__init__()
        import importlib
        from types import SimpleNamespace
        name = {"patchtst": "PatchTST", "itransformer": "iTransformer", "dlinear": "DLinear", "timesnet": "TimesNet"}[cfg.model]
        mod = importlib.import_module(f"pact.tslib.models.{name}")
        c = SimpleNamespace(task_name="long_term_forecast", seq_len=cfg.seq_len, label_len=cfg.label_len,
                            pred_len=cfg.pred_len, enc_in=cfg.num_vars, dec_in=cfg.num_vars, c_out=cfg.num_vars,
                            d_model=cfg.d_model, n_heads=cfg.n_heads, e_layers=cfg.e_layers, d_layers=cfg.d_layers,
                            d_ff=cfg.d_ff, dropout=cfg.dropout, factor=cfg.factor, embed="timeF", freq="h",
                            activation="gelu", top_k=cfg.top_k, num_kernels=cfg.num_kernels, moving_avg=cfg.moving_avg,
                            num_class=1, output_attention=False, use_norm=1)
        self.net = mod.Model(c)

    def forward(self, x, emb=None, x_mark=None):
        return self.net(x, x_mark, None, None), {}


class RLinear(nn.Module):
    """RLinear (Li et al., 2023): RevIN + one linear map shared across variates."""

    def __init__(self, cfg):
        super().__init__()
        self.revin = RevIN(cfg.num_vars, affine=True)
        self.drop = nn.Dropout(cfg.dropout)
        self.linear = nn.Linear(cfg.seq_len, cfg.pred_len)

    def forward(self, x, emb=None, x_mark=None):
        z = self.drop(self.revin.norm(x))
        return self.revin.denorm(self.linear(z.transpose(1, 2)).transpose(1, 2)), {}
