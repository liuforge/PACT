import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

# 导入项目依赖
from pact.legacy.v5_norm import Normalize
from pact.legacy.v5_decomposition import EnhancedDualDecomposition

# ============================================================
#  基础组件模块 (Basic Components)
# ============================================================

class ContrastiveAligner(nn.Module):
    """
    对比学习对齐模块：计算 InfoNCE Loss
    """
    def __init__(self, d_model, temperature=0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.d_model = d_model

    def forward(self, features_a, features_b):
        B, N, C = features_a.shape
        a_norm = F.normalize(features_a, dim=-1)
        b_norm = F.normalize(features_b, dim=-1)
        sim_matrix = torch.matmul(a_norm, b_norm.transpose(-2, -1)) / self.temperature
        labels = torch.arange(N, device=features_a.device)
        loss = F.cross_entropy(sim_matrix.view(B * N, N), labels.repeat(B), reduction='mean')
        return loss

def _build_causal_mask(seq_len: int, device):
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

# ============================================================
#  新增：细节修正模块 (Detail Refinement Module) - 降 MAE 专用
# ============================================================
class LocalDetailRefinement(nn.Module):
    """
    轻量级 CNN 分支，绕过分解模块，直接从原始序列提取高频突变信息。
    结构：1D Conv -> GELU -> 1D Conv (Dilated) -> Projection
    """
    def __init__(self, seq_len, pred_len, channel, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 使用较小的卷积核捕捉瞬时变化
        self.conv_in = nn.Conv1d(channel, channel, kernel_size=3, padding=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # 膨胀卷积扩大感受野，捕捉局部上下文
        self.conv_mid = nn.Conv1d(channel, channel, kernel_size=3, padding=2, dilation=2)
        
        # 简单的线性映射将 seq_len 映射到 pred_len
        self.project = nn.Linear(seq_len, pred_len)
        
        # 最后的微调层
        self.final_conv = nn.Conv1d(channel, channel, kernel_size=1)

    def forward(self, x):
        # x: [B, L, N] -> [B, N, L]
        x = x.permute(0, 2, 1)
        
        res = x
        x = self.conv_in(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.conv_mid(x) + res # 残差连接
        x = self.activation(x)
        
        # 时间维度投影: [B, N, Seq] -> [B, N, Pred]
        x = self.project(x)
        
        x = self.final_conv(x)
        
        # [B, N, Pred] -> [B, Pred, N]
        return x.permute(0, 2, 1)

# ============================================================
#  核心创新模块 (Core Innovation Modules) - V2/V4 Updated
# ============================================================

class PrototypeDynamicPromptLearning(nn.Module):
    """Module 1: PDPL (带正交正则化) - [已修复维度匹配问题]"""
    def __init__(self, d_model, n_prototypes=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_prototypes = n_prototypes
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, d_model))
        nn.init.xavier_uniform_(self.prototypes)
        self.gate_layer = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, ts_features, llm_features):
        # ts_features: [Batch, Num_Nodes, D_model]
        # llm_features: 通常是 [Batch, 1, D_model] 或 [Batch, Num_Nodes, D_model]
        B, N, C = ts_features.shape
        
        scores = torch.matmul(ts_features, self.prototypes.t()) / math.sqrt(C)
        attn_weights = F.softmax(scores, dim=-1)
        dynamic_prompt = torch.matmul(attn_weights, self.prototypes) # [B, N, D]

        # --- FIX START: 维度适配逻辑 ---
        # 如果 llm_features 是全局特征(1个节点)，而我们需要对齐 N 个节点
        if llm_features.dim() == 3 and llm_features.shape[1] == 1 and N > 1:
            # 扩展 llm_features 到 [B, N, D]，使用 expand 避免额外内存开销
            llm_features = llm_features.expand(-1, N, -1)
        # --- FIX END ---

        # 此时 llm_features 和 dynamic_prompt 的形状都是 [B, N, D]，可以安全拼接
        concat_features = torch.cat([llm_features, dynamic_prompt], dim=-1)
        
        gate = self.gate_layer(concat_features)
        enhanced_prompt = gate * dynamic_prompt + (1 - gate) * llm_features
        
        # 正交正则化
        P_norm = F.normalize(self.prototypes, dim=-1)
        cosine_sim = torch.matmul(P_norm, P_norm.t())
        I = torch.eye(self.n_prototypes, device=ts_features.device)
        ortho_loss = torch.norm(cosine_sim - I, p='fro')
        
        return self.norm(enhanced_prompt), ortho_loss

class BiDirectionalIterativeCoAttention(nn.Module):
    """Module 2: BICA (带残差门控)"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn_forward = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_backward = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_final = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff_forward = self._build_ff(d_model, dropout)
        self.ff_backward = self._build_ff(d_model, dropout)
        self.ff_final = self._build_ff(d_model, dropout)
        self.norm_fw1 = nn.LayerNorm(d_model)
        self.norm_fw2 = nn.LayerNorm(d_model)
        self.norm_bw1 = nn.LayerNorm(d_model)
        self.norm_bw2 = nn.LayerNorm(d_model)
        self.norm_fin1 = nn.LayerNorm(d_model)
        self.norm_fin2 = nn.LayerNorm(d_model)
        self.fusion_gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
        
    def _build_ff(self, d_model, dropout):
        return nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))

    def forward(self, ts_features, prompt_features):
        fw_out, _ = self.attn_forward(ts_features, prompt_features, prompt_features)
        h_ts_1 = self.norm_fw1(ts_features + fw_out)
        h_ts_1 = self.norm_fw2(h_ts_1 + self.ff_forward(h_ts_1))
        
        bw_out, _ = self.attn_backward(prompt_features, h_ts_1, h_ts_1)
        p_refined = self.norm_bw1(prompt_features + bw_out)
        p_refined = self.norm_bw2(p_refined + self.ff_backward(p_refined))
        
        fin_out, _ = self.attn_final(h_ts_1, p_refined, p_refined)
        h_final = self.norm_fin1(h_ts_1 + fin_out)
        h_final = self.norm_fin2(h_final + self.ff_final(h_final))
        
        gate_input = torch.cat([ts_features, h_final], dim=-1)
        g = self.fusion_gate(gate_input)
        final_output = g * h_final + (1 - g) * ts_features
        return final_output, (h_ts_1, p_refined)

# ============================================================
#  时间序列编码器 (TS Encoders)
# ============================================================

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, window_size: Optional[int] = None, dropout: float = 0.1, device: str = "cuda"):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.d_head = d_model // n_heads
        self.device = device
        self.qkv_proj = nn.Linear(d_model, 3 * d_model).to(device)
        self.out_proj = nn.Linear(d_model, d_model).to(device)
        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        if self.window_size is not None:
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5) 
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(attn_output)

class ConvTransformerUnifiedEncoder(nn.Module):
    def __init__(self, d_model=64, seq_len=96, transformer_heads=8, dropout=0.15, device="cuda"):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(seq_len, d_model).to(device)
        nn.init.xavier_uniform_(self.input_proj.weight, gain=0.8)
        self.conv_position_enhancer = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.BatchNorm1d(d_model), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model), nn.ReLU(), nn.Dropout(dropout)
        ).to(device)
        self.transformer_encoder = nn.Sequential(
            MultiHeadAttentionLayer(d_model=d_model, n_heads=transformer_heads, window_size=None, dropout=dropout, device=device),
            nn.Linear(d_model, d_model * 3), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model * 3, d_model), nn.LayerNorm(d_model)
        ).to(device)
        self.final_proj = nn.Linear(d_model, d_model).to(device)

    def forward(self, x):
        B, N, L = x.shape
        x_reshaped = x.reshape(B * N, L)
        x_projected = self.input_proj(x_reshaped)
        x_seq = x_projected.unsqueeze(-1).expand(-1, -1, L)
        conv_enhanced = self.conv_position_enhancer(x_seq).transpose(1, 2)
        transformer_output = self.transformer_encoder(conv_enhanced)
        pooled_output = transformer_output.mean(dim=1)
        output = self.final_proj(pooled_output).view(B, N, self.d_model)
        return output

class ConvFFN(nn.Module):
    def __init__(self, d_model, dropout=0.1, expansion_factor=4):
        super().__init__()
        d_ff = d_model * expansion_factor
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(d_ff, d_ff, kernel_size=3, padding=1, groups=d_ff)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout * 0.5)

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.depthwise_conv(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = x.transpose(-1, -2)
        return x

class LightweightNoMaskTransformerEncoder(nn.Module):
    def __init__(self, seq_len, d_model, n_layers=2, n_heads=8, 
                 window_size=None, dropout=0.1, stoch_depth_rate=0.1,
                 pred_len=96, device="cuda", causal_policy: str = "auto"):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.window_size = window_size
        self.device = device
        self.value_embedding = nn.Linear(1, d_model).to(device)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model, device=device) * 0.01)
        self.input_dropout = nn.Dropout(dropout * 0.5).to(device)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout, batch_first=True
        )
        self.encoder_layers = nn.TransformerEncoder(encoder_layer, num_layers=n_layers).to(device)
        self.norm = nn.LayerNorm(d_model).to(device)
        
    def forward(self, x):
        B, N, L = x.shape
        x = x.reshape(B * N, L, 1)
        x = self.value_embedding(x)
        x = x + self.pos_encoding[:, :L, :]
        x = self.input_dropout(x)
        x = self.encoder_layers(x)
        x = x.mean(dim=1) 
        return self.norm(x.reshape(B, N, self.d_model))

class ConvNomaskHybridEncoder(nn.Module):
    def __init__(self, d_model=64, seq_len=96, conv_channels=32, transformer_heads=8, num_scales=3, dropout=0.1, device="cuda"):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.device = device
        self.input_proj = nn.Linear(seq_len, conv_channels).to(device)
        self.feature_adapter = nn.Linear(conv_channels, d_model).to(device)
        self.final_proj = nn.Linear(d_model, d_model).to(device)
        self.conv_block = nn.Sequential(
            nn.Conv1d(conv_channels, conv_channels, 3, padding=1),
            nn.BatchNorm1d(conv_channels), nn.ReLU(), nn.Dropout(dropout)
        ).to(device)
        self.trans_block = nn.TransformerEncoderLayer(d_model, transformer_heads, dim_feedforward=4*d_model, dropout=dropout, batch_first=True).to(device)

    def forward(self, x):
        B, N, L = x.shape
        x_reshaped = x.reshape(B*N, L)
        x_proj = self.input_proj(x_reshaped) 
        x_conv = x_proj.unsqueeze(-1).expand(-1, -1, L)
        x_conv = self.conv_block(x_conv).transpose(1, 2)
        x_conv = self.feature_adapter(x_conv)
        x_trans = self.feature_adapter(x_proj).unsqueeze(1).expand(-1, L, -1)
        x_trans = self.trans_block(x_trans)
        out = x_conv + x_trans
        out = out.mean(dim=1) 
        out = self.final_proj(out).view(B, N, self.d_model)
        return out

# ============================================================
#  主模型 (Main Model) - V5 (Integrated Detail Refinement)
# ============================================================

class Dual_Decomposed(nn.Module):
    def __init__(self, device="cuda", channel=32, num_nodes=7, seq_len=96, pred_len=96, 
                 dropout_n=0.1, d_llm=768, e_layer=1, d_layer=1, head=8, kernel_size=25, 
                 learnable_decomposition=True, ts_encoder_type='conv', disable_patch=False):
        super().__init__()
        self.c_to_length = nn.Linear(channel, pred_len).to(device)
    def forward(self, x, mark, emb):
        return self.c_to_length(x.mean(dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)), 0

class Dual_Decomposed_Advanced(nn.Module):
    """
    V5 模型：集成 PDPL, BICA, 且增加了 LocalDetailRefinement (降 MAE 专用)
    """
    def __init__(
        self,
        device="cuda:0",
        channel=32,
        num_nodes=7,
        seq_len=96,
        pred_len=96,
        dropout_n=0.1,
        d_llm=768,
        e_layer=1,
        d_layer=1,
        head=8,
        kernel_size=25,
        learnable_decomposition=True,
        fusion_strategy="attention",
        ts_encoder_type='conv_transformer_unified',
        n_prototypes=16,
        disable_patch=False,
        nomask_window_size=0,
        nomask_stoch_depth=0.0,
        nomask_dropout=0.0,
        ffn_expansion=0
    ):
        super().__init__()

        self.device = device
        self.channel = channel
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dropout_n = dropout_n
        self.d_llm = d_llm
        self.fusion_strategy = fusion_strategy
        self.ts_encoder_type = ts_encoder_type

        # 1. RevIN
        self.normalize_layers = Normalize(self.num_nodes, affine=True).to(self.device)
        
        # 2. 分解
        self.decomposition = EnhancedDualDecomposition(
            num_features=num_nodes,
            seq_len=seq_len,
            kernel_size=kernel_size,
            learnable=learnable_decomposition,
            pred_len=pred_len
        ).to(self.device)

        # 3. [New] 细节修正分支 (专门降 MAE)
        self.detail_refinement = LocalDetailRefinement(
            seq_len=seq_len, 
            pred_len=pred_len, 
            channel=num_nodes, # 直接处理原始节点维度
            dropout=dropout_n
        ).to(self.device)

        # 4. TS Encoder
        if ts_encoder_type == 'conv_transformer_unified':
            self.ts_encoder = ConvTransformerUnifiedEncoder(
                d_model=self.channel, seq_len=self.seq_len, transformer_heads=head,
                dropout=self.dropout_n, device=self.device
            ).to(self.device)
        elif ts_encoder_type == 'conv_nomask_hybrid':
             self.ts_encoder = ConvNomaskHybridEncoder(
                d_model=self.channel, seq_len=self.seq_len, transformer_heads=head,
                dropout=self.dropout_n, device=self.device
             ).to(self.device)
        elif ts_encoder_type == 'transformer_nomask':
            self.ts_encoder = LightweightNoMaskTransformerEncoder(
                seq_len=self.seq_len, d_model=self.channel, n_layers=e_layer, n_heads=head,
                pred_len=pred_len, device=self.device, dropout=self.dropout_n
            ).to(self.device)
            self.nomask_encoder = self.ts_encoder 
        else:
            self.ts_encoder = nn.Linear(seq_len, channel).to(self.device)

        # 5. LLM Setup
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_llm, nhead=head, batch_first=True, 
            norm_first=True, dropout=self.dropout_n
        ).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(self.prompt_encoder_layer, num_layers=e_layer).to(self.device)
        self.llm_to_channel = nn.Linear(self.d_llm, self.channel).to(self.device)

        # 6. PDPL & BICA
        self.trend_pdpl = PrototypeDynamicPromptLearning(d_model=self.channel, n_prototypes=n_prototypes, dropout=self.dropout_n).to(self.device)
        self.trend_bica = BiDirectionalIterativeCoAttention(d_model=self.channel, n_heads=head, dropout=self.dropout_n).to(self.device)
        self.seasonal_pdpl = PrototypeDynamicPromptLearning(d_model=self.channel, n_prototypes=n_prototypes, dropout=self.dropout_n).to(self.device)
        self.seasonal_bica = BiDirectionalIterativeCoAttention(d_model=self.channel, n_heads=head, dropout=self.dropout_n).to(self.device)
        self.contrastive_aligner = ContrastiveAligner(self.channel).to(self.device)

        # 7. Head
        self.c_to_length = nn.Linear(self.channel, self.pred_len).to(self.device)

        # 8. Fusion
        if fusion_strategy == "attention":
            self.fusion_attention = nn.MultiheadAttention(embed_dim=2, num_heads=1, batch_first=True).to(self.device)
        elif fusion_strategy == "gating":
            self.fusion_gate = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid()).to(self.device)
        elif fusion_strategy == "weighted":
            self.fusion_alpha_logits = nn.Parameter(torch.zeros(self.num_nodes, device=self.device))

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_data, input_data_mark, embeddings):
        input_data = input_data.float()
        embeddings = embeddings.float()
        
        # 1. RevIN Normalize
        norm_data = self.normalize_layers(input_data, 'norm')

        # 2. Decomposition
        # EnhancedDualDecomposition返回 (trend, seasonal)
        trend_data, seasonal_data = self.decomposition(norm_data)
        
        # 3. [New] Calculate Detail/Residual Branch (Bypassing LLM)
        # 输入是归一化后的数据，直接提取高频修正项
        detail_pred = self.detail_refinement(norm_data) # [B, Pred, N]

        # 4. LLM Embedding Process
        emb = embeddings
        if emb.dim() == 4: emb = emb.squeeze(-1)
        if emb.shape[1] == self.d_llm: emb = emb.permute(0, 2, 1)
        emb = self.prompt_encoder(emb)
        emb = self.llm_to_channel(emb)
        
        # 5. Main Branches
        trend_pred, trend_loss = self._process_branch(trend_data, emb, 'trend')
        seasonal_pred, seasonal_loss = self._process_branch(seasonal_data, emb, 'seasonal')
        
        # 6. Fusion (Trend + Seasonal)
        coarse_pred = self._fuse_predictions(trend_pred, seasonal_pred)
        
        # 7. [New] Add Detail Refinement
        # 最终预测 = (Trend+Seasonal) + Detail_CNN
        final_pred = coarse_pred + detail_pred

        # 8. Denormalize
        final_pred = self.normalize_layers(final_pred, 'denorm')

        total_loss = 0
        if trend_loss: total_loss += trend_loss
        if seasonal_loss: total_loss += seasonal_loss

        return final_pred, total_loss

    def _process_branch(self, branch_data, embeddings, branch_type='trend'):
        branch_data_perm = branch_data.permute(0, 2, 1)
        enc_out = self.ts_encoder(branch_data_perm)
        
        if branch_type == 'trend':
            pdpl, bica = self.trend_pdpl, self.trend_bica
        else:
            pdpl, bica = self.seasonal_pdpl, self.seasonal_bica

        enhanced_prompt, ortho_loss = pdpl(enc_out, embeddings)
        final_out, (h_ts_1, p_refined) = bica(enc_out, enhanced_prompt)

        loss_val = None
        if self.training:
            loss_val = self.contrastive_aligner(h_ts_1, p_refined) * 0.1 + ortho_loss * 0.05

        pred_out = self.c_to_length(final_out).permute(0, 2, 1)
        return pred_out, loss_val

    def _fuse_predictions(self, trend_pred, seasonal_pred):
        if self.fusion_strategy == "simple": return trend_pred + seasonal_pred
        elif self.fusion_strategy == "attention":
            combined = torch.stack([trend_pred, seasonal_pred], dim=-1)
            B, L, N = trend_pred.shape
            combined_reshaped = combined.view(B * N, L, 2)
            attended, _ = self.fusion_attention(combined_reshaped, combined_reshaped, combined_reshaped)
            attended = attended.view(B, N, L, 2).permute(0, 2, 1, 3)
            weights = torch.softmax(attended, dim=-1)
            return weights[..., 0] * trend_pred + weights[..., 1] * seasonal_pred
        elif self.fusion_strategy == "gating":
            combined = torch.stack([trend_pred, seasonal_pred], dim=-1)
            gate = self.fusion_gate(combined).squeeze(-1)
            return gate * trend_pred + (1 - gate) * seasonal_pred
        elif self.fusion_strategy == "weighted":
            alpha = torch.sigmoid(self.fusion_alpha_logits).view(1, 1, self.num_nodes)
            return alpha * trend_pred + (1 - alpha) * seasonal_pred
        return trend_pred + seasonal_pred

# Loss (FreqMAE V4.1)
class FreqMAE(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        pred_fft = torch.fft.rfft(pred, dim=1, norm='ortho')
        target_fft = torch.fft.rfft(target, dim=1, norm='ortho')
        loss_real = torch.abs(pred_fft.real - target_fft.real).mean()
        loss_imag = torch.abs(pred_fft.imag - target_fft.imag).mean()
        return loss_real + loss_imag