"""PACT: Prototype Alignment and Co-attention for Time series.

Pipeline (each step can be switched off for ablations):
  RevIN -> Gaussian trend/seasonal decomposition -> shared per-variate encoder
  -> PDPL (prototype retrieval + knowledge gate with the LLM anchor)
  -> BDCA (bi-directional co-attention + residual gate) -> per-branch linear head -> fusion -> RevIN^-1

Token layout. Every attention module works on tensors of shape [S, T, D] (S sequences of T tokens):
  * token_mode="patch":   S = B*N, T = number of patches  (attention along time, per variate)
  * token_mode="variate": S = B,   T = N                  (attention across variates, iTransformer-like)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    def __init__(self, num_vars, affine=True, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_vars))
            self.bias = nn.Parameter(torch.zeros(num_vars))

    def norm(self, x):  # x [B, L, N]
        self.mean = x.mean(1, keepdim=True).detach()
        self.std = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + self.eps).detach()
        x = (x - self.mean) / self.std
        return x * self.weight + self.bias if self.affine else x

    def denorm(self, y):  # y [B, H, N]
        if self.affine:
            y = (y - self.bias) / (self.weight + self.eps * self.eps)
        return y * self.std + self.mean


class SeriesDecomp(nn.Module):
    """Per-variate smoothing filter: trend = conv(x, w), seasonal = x - trend.

    mode="gaussian"       learnable bandwidth sigma = softplus(s_hat)   (Eq. 1 of the paper)
    mode="gaussian_fixed" Gaussian kernel with fixed sigma
    mode="moving_avg"     uniform kernel (Autoformer/DLinear decomposition)
    """

    def __init__(self, kernel_size=25, mode="gaussian", init_sigma=None):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.kernel_size = kernel_size
        self.mode = mode
        sigma = init_sigma if init_sigma is not None else kernel_size / 6.0
        raw = math.log(math.expm1(sigma))  # inverse softplus
        if mode == "gaussian":
            self.sigma_hat = nn.Parameter(torch.tensor(raw))
        else:
            self.register_buffer("sigma_hat", torch.tensor(raw))
        self.register_buffer("pos", torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2)

    @property
    def sigma(self):
        return F.softplus(self.sigma_hat)

    def kernel(self):
        if self.mode == "moving_avg":
            return torch.full_like(self.pos, 1.0 / self.kernel_size)
        w = torch.exp(-self.pos ** 2 / (2 * self.sigma ** 2))
        return w / w.sum()

    def forward(self, x):  # x [B, L, N]
        B, L, N = x.shape
        z = x.permute(0, 2, 1).reshape(B * N, 1, L)
        pad = (self.kernel_size - 1) // 2
        z = F.pad(z, (pad, pad), mode="replicate")
        trend = F.conv1d(z, self.kernel().view(1, 1, -1)).reshape(B, N, L).permute(0, 2, 1)
        return trend, x - trend


def _transformer(d_model, n_heads, d_ff, dropout, n_layers):
    layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)


class PatchEncoder(nn.Module):
    """Shared per-variate encoder: patch embedding -> depth-wise temporal conv -> Transformer."""

    def __init__(self, seq_len, patch_len, stride, d_model, n_heads, e_layers, d_ff, dropout):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.pad_end = stride if patch_len > 1 else 0
        self.num_tokens = (seq_len + self.pad_end - patch_len) // stride + 1
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.num_tokens, d_model) * 0.02)
        self.local = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = _transformer(d_model, n_heads, d_ff, dropout, e_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):  # x [B, L, N] -> [B*N, P, D]
        B, L, N = x.shape
        z = x.permute(0, 2, 1).reshape(B * N, 1, L)
        if self.pad_end:
            z = F.pad(z, (0, self.pad_end), mode="replicate")
        z = z.unfold(-1, self.patch_len, self.stride).squeeze(1)  # [B*N, P, patch_len]
        h = self.drop(self.embed(z) + self.pos)
        h = h + self.local(h.transpose(1, 2)).transpose(1, 2)
        return self.norm(self.layers(h))


class VariateEncoder(nn.Module):
    """Each variate's whole look-back window becomes one token (iTransformer-style)."""

    def __init__(self, seq_len, d_model, n_heads, e_layers, d_ff, dropout):
        super().__init__()
        self.embed = nn.Linear(seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = _transformer(d_model, n_heads, d_ff, dropout, e_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):  # x [B, L, N] -> [B, N, D]
        return self.norm(self.layers(self.drop(self.embed(x.permute(0, 2, 1)))))


class PromptAdapter(nn.Module):
    """Frozen-LLM embedding [B, N, d_llm] -> anchor [B, N, D] through a bottleneck Transformer."""

    def __init__(self, d_llm, d_bottleneck, d_model, n_heads, dropout):
        super().__init__()
        self.inp_norm = nn.LayerNorm(d_llm)
        self.down = nn.Linear(d_llm, d_bottleneck)
        self.mix = _transformer(d_bottleneck, n_heads, 4 * d_bottleneck, dropout, 1)
        self.up = nn.Linear(d_bottleneck, d_model)

    def forward(self, e):
        return self.up(self.mix(self.down(self.inp_norm(e))))


class PDPL(nn.Module):
    """Prototype-based Dynamic Prompt Learning: soft retrieval over a learnable codebook + knowledge gate.

    mode="dot": keys = values = P, scaled dot product (first version; E_dyn is a smooth projection of H)
    mode="kv":  separate key and value prototypes (a key-value memory over the training set), cosine
                retrieval with a learnable temperature, so the prompt can carry information H does not contain
    """

    def __init__(self, d_model, n_prototypes, mode="dot", temp_init=10.0):
        super().__init__()
        self.mode = mode
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, d_model))  # keys (and values for "dot")
        if mode == "kv":
            self.values = nn.Parameter(torch.randn(n_prototypes, d_model))
            self.log_scale = nn.Parameter(torch.tensor(math.log(temp_init)))
        elif mode != "dot":
            raise ValueError(mode)
        self.gate = nn.Linear(2 * d_model, d_model)
        self.last = {}

    def forward(self, h, anchor=None):  # h, anchor [S, T, D]
        if self.mode == "kv":
            sim = F.normalize(h, dim=-1) @ F.normalize(self.prototypes, dim=-1).t()
            attn = torch.softmax(sim * self.log_scale.exp(), dim=-1)  # [S, T, K]
            e_dyn = attn @ self.values
        else:
            attn = torch.softmax(h @ self.prototypes.t() / math.sqrt(h.shape[-1]), dim=-1)
            e_dyn = attn @ self.prototypes
        with torch.no_grad():  # diagnostics: normalised retrieval entropy (1 = uniform) and prototype usage
            ent = -(attn * attn.clamp_min(1e-9).log()).sum(-1).mean() / math.log(attn.shape[-1])
            self.last = {"attn_entropy": float(ent), "usage_max": float(attn.mean((0, 1)).max())}
        if anchor is None:
            return e_dyn, attn
        lam = torch.sigmoid(self.gate(torch.cat([anchor, e_dyn], dim=-1)))
        self.last["gate_dyn"] = float(lam.detach().mean())  # share of the dynamic prompt vs. the LLM anchor
        return lam * e_dyn + (1 - lam) * anchor, attn

    def diversity_loss(self):
        p = F.normalize(self.prototypes, dim=-1)
        eye = torch.eye(p.shape[0], device=p.device)
        return ((p @ p.t() - eye) ** 2).mean()


class CrossBlock(nn.Module):
    """Post-LN residual cross-attention (+ optional position-wise FFN)."""

    def __init__(self, d_model, n_heads, dropout, ffn=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * d_model, d_model), nn.Dropout(dropout)) if ffn else None
        self.norm2 = nn.LayerNorm(d_model) if ffn else None

    def forward(self, q, kv):
        q = self.norm1(q + self.drop(self.attn(q, kv, kv, need_weights=False)[0]))
        if self.ffn is not None:
            q = self.norm2(q + self.ffn(q))
        return q


class BDCA(nn.Module):
    """Bi-Directional Co-Attention.

    For i = 1..I:  H <- Fwd_i(H, E)  (forward retrieval);  E <- Bwd_i(E, H)  (backward calibration)
    then a final forward retrieval H <- Fwd_{I+1}(H, E) so the last calibrated prompt is used,
    followed by the residual gate  g * H + (1 - g) * H0.
    iters=0 reduces to a single (uni-directional) cross-attention block, i.e. the "w/o BDCA" variant.

    cross_variate=True (patch layout): after the temporal calibration, the prompt tokens of every variate
    additionally attend to the summaries (time-averaged features) of all variates of the same sample, so
    the backward step injects inter-variate dependencies that the channel-independent encoder lacks.
    """

    def __init__(self, d_model, n_heads, iters, dropout, ffn=True, cross_variate=False):
        super().__init__()
        self.iters = iters
        self.cross_variate = cross_variate
        self.fwd = nn.ModuleList([CrossBlock(d_model, n_heads, dropout, ffn) for _ in range(iters + 1)])
        self.bwd = nn.ModuleList([CrossBlock(d_model, n_heads, dropout, ffn) for _ in range(iters)])
        if cross_variate:
            self.bwd_var = nn.ModuleList([CrossBlock(d_model, n_heads, dropout, ffn) for _ in range(iters)])
        self.gate = nn.Linear(2 * d_model, d_model)
        self.last = {}

    def _variate_calibration(self, i, e, h, B):
        S, T, D = e.shape
        N = S // B
        summary = h.mean(1).reshape(B, N, D)  # one token per variate
        e = self.bwd_var[i](e.reshape(B, N * T, D), summary)
        return e.reshape(S, T, D)

    def forward(self, h0, e0, batch_size=None):
        h, e = h0, e0
        for i in range(self.iters):
            h = self.fwd[i](h, e)
            e = self.bwd[i](e, h)
            if self.cross_variate and batch_size is not None:
                e = self._variate_calibration(i, e, h, batch_size)
        h = self.fwd[-1](h, e)
        g = torch.sigmoid(self.gate(torch.cat([h0, h], dim=-1)))
        self.last = {"gate_bdca": float(g.detach().mean())}  # share of the BDCA output vs. the encoder feature
        return g * h + (1 - g) * h0


def token_infonce(h, e, log_tau):
    """Token-wise InfoNCE: same token is the positive, other tokens of the same sequence are negatives."""
    S, T, _ = h.shape
    if T < 2:
        return h.new_zeros(())
    tau = log_tau.exp().clamp(min=0.01)
    logits = F.normalize(h, dim=-1) @ F.normalize(e, dim=-1).transpose(1, 2) / tau  # [S, T, T]
    labels = torch.arange(T, device=h.device).repeat(S)
    return F.cross_entropy(logits.reshape(S * T, T), labels)


class PACT(nn.Module):
    LLM_MODES = ("stored", "shuffled", "learned", "random", "none")

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        N, L, H, D = cfg.num_vars, cfg.seq_len, cfg.pred_len, cfg.d_model
        self.revin = RevIN(N, affine=cfg.revin_affine) if cfg.revin else None
        self.decomp = None if cfg.decomp == "none" else SeriesDecomp(cfg.kernel_size, cfg.decomp, cfg.init_sigma)
        self.branches = ["trend", "seasonal"] if self.decomp is not None else ["main"]

        if cfg.token_mode == "patch":
            self.encoder = PatchEncoder(L, cfg.patch_len, cfg.stride, D, cfg.n_heads, cfg.e_layers, cfg.d_ff, cfg.dropout)
            head_in = self.encoder.num_tokens * D
        elif cfg.token_mode == "variate":
            self.encoder = VariateEncoder(L, D, cfg.n_heads, cfg.e_layers, cfg.d_ff, cfg.dropout)
            head_in = D
        else:
            raise ValueError(cfg.token_mode)

        # LLM semantic anchor
        if cfg.llm not in self.LLM_MODES:
            raise ValueError(cfg.llm)
        self.use_llm = cfg.llm != "none"
        if self.use_llm:
            self.adapter = PromptAdapter(cfg.d_llm, cfg.adapter_dim, D, cfg.adapter_heads, cfg.dropout)
            if cfg.llm == "learned":  # capacity-matched control: same adapter, no pre-trained input
                self.llm_vec = nn.Parameter(torch.randn(N, cfg.d_llm) * 0.02)
            elif cfg.llm == "random":  # fixed random vectors with no sample information
                g = torch.Generator().manual_seed(cfg.seed)
                self.register_buffer("llm_vec", torch.randn(N, cfg.d_llm, generator=g))

        self.pdpl = PDPL(D, cfg.n_prototypes, cfg.pdpl_mode, cfg.pdpl_temp) if cfg.pdpl else None
        self.has_prompt = self.use_llm or self.pdpl is not None
        use_bdca = self.has_prompt and cfg.bdca_iters >= 0
        # trend_head="linear": the trend branch is a per-variate linear map (L -> H); only the seasonal branch
        # goes through the encoder, PDPL and BDCA (division of labour between the two streams)
        self.linear_trend = self.decomp is not None and cfg.trend_head == "linear"
        deep = [b for b in self.branches if not (self.linear_trend and b == "trend")]
        xvar = bool(cfg.bdca_xvar) and cfg.token_mode == "patch"
        self.bdca = nn.ModuleDict({b: BDCA(D, cfg.n_heads, cfg.bdca_iters, cfg.dropout, cfg.bdca_ffn, xvar)
                                   for b in deep}) if use_bdca else None
        self.heads = nn.ModuleDict({b: nn.Sequential(nn.Dropout(cfg.head_dropout), nn.Linear(head_in, H))
                                    for b in deep})
        if self.linear_trend:
            self.trend_linear = nn.Linear(L, H)
        if self.decomp is not None:
            if cfg.fusion == "scalar":
                self.fuse_w = nn.Parameter(torch.zeros(1))
            elif cfg.fusion == "variate":
                self.fuse_w = nn.Parameter(torch.zeros(N))
            elif cfg.fusion != "sum":
                raise ValueError(cfg.fusion)
        if self.has_prompt and cfg.lambda_align > 0:
            self.log_tau = nn.Parameter(torch.tensor(math.log(cfg.tau_init)))

    def diagnostics(self):
        """Last-batch internal statistics of PDPL and BDCA (attention entropy, gate openings)."""
        out = {}
        if self.pdpl is not None:
            out.update({f"pdpl_{k}": v for k, v in self.pdpl.last.items()})
        if self.bdca is not None:
            for b, m in self.bdca.items():
                out.update({f"{b}_{k}": v for k, v in m.last.items()})
        if self.decomp is not None and self.cfg.fusion != "sum":
            out["fuse_trend_weight"] = float(torch.sigmoid(self.fuse_w).mean())
        return out

    def _anchor(self, emb, B):
        """[B, N, D] anchor from stored embeddings or from the control vectors."""
        if self.cfg.llm in ("learned", "random"):
            emb = self.llm_vec.unsqueeze(0).expand(B, -1, -1)
        return self.adapter(emb)

    def _to_tokens(self, a, B, T):  # [B, N, D] -> [S, T, D]
        if self.cfg.token_mode == "patch":
            N, D = a.shape[1], a.shape[2]
            return a.reshape(B * N, 1, D).expand(-1, T, -1)
        return a

    def _head(self, h, branch, B, N):
        if self.cfg.token_mode == "patch":
            y = self.heads[branch](h.reshape(B * N, -1)).reshape(B, N, -1)
        else:
            y = self.heads[branch](h)
        return y.permute(0, 2, 1)  # [B, H, N]

    def forward(self, x, emb=None, x_mark=None, return_details=False):
        """x [B, L, N] (dataset-scaled); emb [B, N, d_llm] for llm in {stored, shuffled}."""
        B, _, N = x.shape
        z = self.revin.norm(x) if self.revin is not None else x
        parts = dict(zip(self.branches, self.decomp(z))) if self.decomp is not None else {"main": z}
        anchor = self._anchor(emb, B) if self.use_llm else None

        aux = {}
        details = {}
        preds = {}
        for b, xb in parts.items():
            if self.linear_trend and b == "trend":
                preds[b] = self.trend_linear(xb.transpose(1, 2)).transpose(1, 2)  # [B, H, N]
                continue
            h0 = self.encoder(xb)  # [S, T, D]
            T = h0.shape[1]
            a = self._to_tokens(anchor, B, T) if anchor is not None else None
            if self.pdpl is not None:
                e, attn = self.pdpl(h0, a)
                details[f"proto_attn_{b}"] = attn
            else:
                e = a
            h = self.bdca[b](h0, e, B if self.cfg.token_mode == "patch" else None) if self.bdca is not None else h0
            if self.training and e is not None and self.cfg.lambda_align > 0:
                aux[f"align_{b}"] = token_infonce(h0, e, self.log_tau)
            preds[b] = self._head(h, b, B, N)

        if self.decomp is None:
            y = preds["main"]
        elif self.cfg.fusion == "sum":
            y = preds["trend"] + preds["seasonal"]
        else:
            w = torch.sigmoid(self.fuse_w)
            y = w * preds["trend"] + (1 - w) * preds["seasonal"]

        if self.training and self.pdpl is not None and self.cfg.lambda_div > 0:
            aux["div"] = self.pdpl.diversity_loss()
        y = self.revin.denorm(y) if self.revin is not None else y
        if return_details:
            return y, aux, details
        return y, aux
