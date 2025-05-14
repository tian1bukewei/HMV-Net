import torch
import torch.nn as nn
import numpy as np


class SEU(nn.Module):
    """
    Semantic Enhancement Unit (SEU): A lightweight, fully differentiable
    refinement module that selectively modulates the depth map using
    geometry-informed saliency priors.
    """

    def __init__(self, beta=0.3, tau=0.5):
        """
        Initialize the Semantic Enhancement Unit

        Parameters:
        -----------
        beta : float
            Contrast strength for residual modulation (default: 0.3)
        tau : float
            Binarization threshold for foreground mask generation (default: 0.5)
        """
        super(SEU, self).__init__()
        self.beta = beta
        self.tau = tau

    def forward(self, depth):
        """
        Forward pass implementing the three-stage SEU refinement:
        1. Saliency prior estimation
        2. Confidence-based mask generation
        3. Residual contrast modulation

        Parameters:
        -----------
        depth : torch.Tensor
            Input depth map tensor

        Returns:
        --------
        torch.Tensor
            Enhanced depth map with improved structural fidelity
        """
        # Normalize and invert depth (Eq. 6)
        depth_min = depth.min()
        depth_max = depth.max()
        D_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)

        # Compute foreground confidence prior
        F = 1.0 - D_norm  # Higher values represent closer objects

        # Thresholding - generate binary foreground mask (Eq. 7)
        M = (F > self.tau).float()

        # Apply residual contrast-based correction (Eq. 8)
        D_refined = depth - self.beta * M * (1.0 - F)

        return D_refined

    def enhance_numpy(self, depth_np):
        """
        NumPy implementation of SEU for inference

        Parameters:
        -----------
        depth_np : numpy.ndarray
            Input depth map as numpy array

        Returns:
        --------
        numpy.ndarray
            Enhanced depth map with improved structural fidelity
        """
        # Normalize and invert depth (Eq. 6)
        depth_min = depth_np.min()
        depth_max = depth_np.max()
        D_norm = (depth_np - depth_min) / (depth_max - depth_min + 1e-8)

        # Compute foreground confidence prior
        F = 1.0 - D_norm  # Higher values represent closer objects

        # Thresholding - generate binary foreground mask (Eq. 7)
        M = (F > self.tau).astype(np.float32)

        # Apply residual contrast-based correction (Eq. 8)
        D_refined = depth_np - self.beta * M * (1.0 - F)

        return D_refined