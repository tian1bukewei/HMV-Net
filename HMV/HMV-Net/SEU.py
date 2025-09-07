import torch
import torch.nn as nn
import numpy as np


class SEU(nn.Module):
    """
    Semantic Enhancement Unit (SEU): A complementary component that leverages
    ERAM's attention results to perform adaptive enhancement on focused regions.

    This module implements lightweight post-processing on attention-identified
    critical regions through depth-guided residual correction.
    """

    def __init__(self, beta=0.3, tau=0.5):
        """
        Initialize the Semantic Enhancement Unit

        Parameters:
        -----------
        beta : float
            Enhancement strength control factor (β in Eq. 10)
            Empirically determined for stable boundary enhancement
        tau : float
            Binary foreground mask threshold (τ in Eq. 9)
            Derived from bimodal distribution analysis
        """
        super(SEU, self).__init__()
        self.beta = beta
        self.tau = tau

    def forward(self, depth, attention_map):
        """
        Forward pass implementing spatially-weighted residual adjustment

        Parameters:
        -----------
        depth : torch.Tensor
            Input depth map tensor D
        attention_map : torch.Tensor, optional
            Attention results from ERAM (Att)

        Returns:
        --------
        torch.Tensor
            Enhanced depth map D_refined with improved boundary precision
        """
        # Normalize depth for confidence computation
        depth_min = depth.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        depth_max = depth.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        D_inv = 1.0 - (depth - depth_min) / (depth_max - depth_min + 1e-8)

        if attention_map is not None:
            # P = normalize(Att ⊙ D_inv)
            P = attention_map * D_inv
            P = P / (P.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8)
        else:
            # Use depth-based confidence only
            P = D_inv

        # M = 𝟙[P > τ]
        M = (P > self.tau).float()

        # Compute depth gradient for structure-aware enhancement
        if len(depth.shape) == 4:  # Batch processing
            # Sobel-like gradient approximation
            grad_h = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])
            grad_w = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1])
            # Pad to maintain spatial dimensions
            grad_h = torch.nn.functional.pad(grad_h, (0, 0, 0, 1))
            grad_w = torch.nn.functional.pad(grad_w, (0, 0, 0, 1))
            grad_D = (grad_h + grad_w) / 2.0
        else:
            # Simple gradient for 2D input
            grad_D = torch.zeros_like(depth)

        D_refined = depth + self.beta * M * (1.0 - P) * grad_D

        return D_refined

    def forward_simple(self, depth):
        """
        Simplified forward pass with ERAM attention
        (Compatible with original implementation)

        Parameters:
        -----------
        depth : torch.Tensor
            Input depth map tensor

        Returns:
        --------
        torch.Tensor
            Enhanced depth map
        """
        # Normalize and invert depth
        depth_min = depth.min()
        depth_max = depth.max()
        D_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)

        # Compute foreground confidence prior
        F = 1.0 - D_norm

        M = (F > self.tau).float()

        # Apply residual correction (simplified version)
        D_refined = depth - self.beta * M * (1.0 - F)

        return D_refined