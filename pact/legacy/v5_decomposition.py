import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

class EnhancedDualDecomposition(nn.Module):
    """
    增强的双分解：在trend+seasonal框架内提升分解质量
    核心思想：通过约束优化让分解更精准
    """
    def __init__(
        self, 
        num_features: int,
        seq_len: int, 
        kernel_size: int = 25,
        n_iterations: int = 3,
        learnable: bool = False,
        pred_len: int = None,
        enable_constraints: bool = None,
    ):
        super(EnhancedDualDecomposition, self).__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.n_iterations = n_iterations
        self.learnable = learnable
        # 约束开关（若为 None 将由 pred_len 自适应决定）
        self.enable_constraints = enable_constraints
        
        # 按预测步长进行自适应门控（短步退化，长步增强）
        # 仅当提供了 pred_len 时生效
        if pred_len is not None:
            if pred_len <= 4:
                # 短期：退化为单层分解；关闭约束
                self.n_iterations = 1
                if self.enable_constraints is None:
                    self.enable_constraints = False
            elif pred_len <= 48:
                # 中期：两层精化；默认关闭约束（防过度平滑）
                self.n_iterations = max(self.n_iterations, 2)
                if self.enable_constraints is None:
                    self.enable_constraints = False
            else:
                # 长期：三层精化；开启约束
                self.n_iterations = max(self.n_iterations, 3)
                if self.enable_constraints is None:
                    self.enable_constraints = True

        # 初始分解核（保守）
        self.initial_conv = self._create_gaussian_conv(kernel_size)
        
        # 精化分解核（渐进式）
        self.refine_convs = nn.ModuleList([
            self._create_gaussian_conv(max(7, kernel_size - i*6)) 
            for i in range(1, self.n_iterations)
        ])
        
        # 分解质量评估
        self.quality_assessor = DecompositionQualityAssessor(seq_len)
        
        # 分解约束权重
        self.smoothness_weight = 0.1
        self.periodicity_weight = 0.05
        
    def _create_gaussian_conv(self, kernel_size: int) -> nn.Conv1d:
        """创建高斯卷积核"""
        padding = kernel_size // 2
        conv = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, 
                        padding=padding, padding_mode='replicate', bias=True)
        
        kernel_size_half = kernel_size // 2
        sigma = kernel_size / 6.0
        weights = torch.zeros(1, 1, kernel_size)
        for i in range(kernel_size):
            weights[0, 0, i] = math.exp(-((i - kernel_size_half) / (2 * sigma)) ** 2)
        
        conv.weight.data = F.softmax(weights, dim=-1)
        conv.bias.data.fill_(0.0)
        
        if not self.learnable:
            conv.weight.requires_grad = False
            conv.bias.requires_grad = False
        
        return conv
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        增强的双分解forward
        """
        # 转换为 [B, N, L] 格式
        x = x.permute(0, 2, 1)  # [B, N, L]
        batch_size, num_features, seq_len = x.shape
        
        # Step 1: 初始分解
        trend, seasonal = self._initial_decomposition(x)
        
        # Step 2: 迭代精化分解
        for i in range(self.n_iterations - 1):
            trend, seasonal = self._refine_decomposition(x, trend, seasonal, i)
        
        # Step 3: 约束优化（可选）
        if self.training and bool(self.enable_constraints):  # 仅训练时优化，推理时保持快速
            trend, seasonal = self._constrained_optimization(x, trend, seasonal)

        # 运行时打印一次关键信息（仅首次调用）
        if not hasattr(self, "_printed_cfg"):
            try:
                print(f"[EnhancedDualDecomposition] n_iterations={self.n_iterations}, "
                      f"constraints={bool(self.enable_constraints)}, kernel_size={self.kernel_size}")
            except Exception:
                pass
            self._printed_cfg = True
        
        # 转换回 [B, L, N] 格式
        trend = trend.permute(0, 2, 1)  # [B, L, N]
        seasonal = seasonal.permute(0, 2, 1)  # [B, L, N]
        
        return trend, seasonal
    
    def _initial_decomposition(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """初始分解：保守的粗分解"""
        trend_outputs = []
        
        for i in range(self.num_features):
            feature_series = x[:, i:i+1, :]  # [B, 1, L]
            trend_feature = self.initial_conv(feature_series)
            trend_outputs.append(trend_feature)
        
        trend = torch.cat(trend_outputs, dim=1)  # [B, N, L]
        seasonal = x - trend
        
        return trend, seasonal
    
    def _refine_decomposition(
        self, 
        x: torch.Tensor, 
        trend: torch.Tensor, 
        seasonal: torch.Tensor, 
        iteration: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        精化分解：从seasonal中进一步提取trend成分
        """
        refine_conv = self.refine_convs[iteration]
        
        # 计算当前分解质量
        quality_score = self.quality_assessor(x, trend, seasonal)
        
        # 如果质量已经很好，跳过精化
        if quality_score > 0.95:
            return trend, seasonal
        
        trend_refinements = []
        
        for i in range(self.num_features):
            seasonal_feature = seasonal[:, i:i+1, :]  # [B, 1, L]
            
            # 从seasonal中提取残余趋势
            additional_trend = refine_conv(seasonal_feature)
            
            # 自适应权重：根据质量评分调整
            refinement_weight = (1.0 - quality_score) * 0.3
            additional_trend = additional_trend * refinement_weight
            
            trend_refinements.append(additional_trend)
        
        # 更新分解
        additional_trend_total = torch.cat(trend_refinements, dim=1)
        trend_refined = trend + additional_trend_total
        seasonal_refined = seasonal - additional_trend_total
        
        return trend_refined, seasonal_refined
    
    def _constrained_optimization(
        self, 
        x: torch.Tensor, 
        trend: torch.Tensor, 
        seasonal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        约束优化：基于平滑性和周期性约束微调分解
        """
        # 计算约束损失
        smoothness_loss = self._compute_smoothness_loss(trend)
        periodicity_loss = self._compute_periodicity_loss(seasonal)
        reconstruction_loss = F.mse_loss(trend + seasonal, x)
        
        # 如果约束已经满足，跳过优化
        total_constraint = (self.smoothness_weight * smoothness_loss + 
                          self.periodicity_weight * periodicity_loss)
        
        if total_constraint < 0.01:  # 约束已满足
            return trend, seasonal
        
        # 轻微调整以满足约束
        # 增强趋势平滑性
        trend_smoothed = self._apply_smoothness_constraint(trend)
        
        # 增强季节性周期性  
        seasonal_enhanced = self._apply_periodicity_constraint(seasonal)
        
        # 保持重构一致性
        residual_error = x - (trend_smoothed + seasonal_enhanced)
        trend_final = trend_smoothed + 0.5 * residual_error  # 趋势承担一半误差
        seasonal_final = seasonal_enhanced + 0.5 * residual_error  # 季节性承担一半误差
        
        return trend_final, seasonal_final
    
    def _compute_smoothness_loss(self, trend: torch.Tensor) -> torch.Tensor:
        """计算趋势平滑性损失"""
        # 二阶差分
        diff2 = trend[:, :, 2:] - 2 * trend[:, :, 1:-1] + trend[:, :, :-2]
        return torch.mean(diff2 ** 2)
    
    def _compute_periodicity_loss(self, seasonal: torch.Tensor) -> torch.Tensor:
        """计算季节性周期性损失"""
        # 简单的自相关度量
        seq_len = seasonal.size(-1)
        
        # 计算不同lag的自相关
        autocorr_loss = 0
        for lag in [1, 7, 24]:  # 常见周期
            if lag < seq_len:
                shifted = F.pad(seasonal[:, :, lag:], (0, lag))
                correlation = F.cosine_similarity(
                    seasonal.flatten(0, 1), 
                    shifted.flatten(0, 1), 
                    dim=-1
                ).mean()
                autocorr_loss += (1 - correlation) ** 2
        
        return autocorr_loss
    
    def _apply_smoothness_constraint(self, trend: torch.Tensor) -> torch.Tensor:
        """应用平滑性约束"""
        # 简单的移动平均平滑
        kernel = torch.ones(1, 1, 5, device=trend.device) / 5
        trend_padded = F.pad(trend, (2, 2), mode='replicate')
        
        smoothed_outputs = []
        for i in range(trend.size(1)):
            feature_trend = trend_padded[:, i:i+1, :]
            smoothed = F.conv1d(feature_trend, kernel)
            smoothed_outputs.append(smoothed)
        
        smoothed_trend = torch.cat(smoothed_outputs, dim=1)
        
        # 轻微混合原始和平滑版本
        return 0.9 * trend + 0.1 * smoothed_trend
    
    def _apply_periodicity_constraint(self, seasonal: torch.Tensor) -> torch.Tensor:
        """应用周期性约束"""
        # FFT增强周期性
        fft = torch.fft.rfft(seasonal, dim=-1)
        magnitude = torch.abs(fft)
        
        # 找到最强的几个频率成分
        top_k = min(5, magnitude.size(-1))
        _, top_indices = torch.topk(magnitude, k=top_k, dim=-1)
        
        # 创建周期性增强掩码
        mask = torch.zeros_like(fft, dtype=torch.float32)
        for i in range(top_k):
            mask.scatter_(-1, top_indices[..., i:i+1], 1.0)
        
        # 增强周期性成分
        enhanced_fft = fft * (1 + 0.1 * mask)
        enhanced_seasonal = torch.fft.irfft(enhanced_fft, n=seasonal.size(-1), dim=-1)
        
        return enhanced_seasonal

class DecompositionQualityAssessor(nn.Module):
    """分解质量评估器"""
    def __init__(self, seq_len: int):
        super(DecompositionQualityAssessor, self).__init__()
        self.seq_len = seq_len
    
    def forward(
        self, 
        original: torch.Tensor, 
        trend: torch.Tensor, 
        seasonal: torch.Tensor
    ) -> torch.Tensor:
        """
        评估分解质量 [0, 1]，1表示完美分解
        """
        # 重构误差
        reconstruction = trend + seasonal
        reconstruction_error = F.mse_loss(reconstruction, original)
        reconstruction_score = torch.exp(-reconstruction_error * 10)  # 转换为[0,1]
        
        # 趋势平滑度
        trend_diff2 = trend[:, :, 2:] - 2 * trend[:, :, 1:-1] + trend[:, :, :-2]
        smoothness_score = torch.exp(-torch.mean(trend_diff2 ** 2) * 100)
        
        # 综合评分
        quality_score = 0.7 * reconstruction_score + 0.3 * smoothness_score
        
        return quality_score.item()
