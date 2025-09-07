import torch
import torch.nn as nn


class ERAM(nn.Module):
    """
    Energy-based Residual Attention Module (ERAM): Implements spatially-adaptive
    feature modulation during training through energy minimization principles while
    reducing to identity mapping during inference.

    During training: Applies energy-based spatial attention to identify and enhance
                    foreground-critical regions through residual connections.
    At inference: Reduces to identity transformation, ensuring zero computational overhead.

    Reference: Modified from SimAM (https://arxiv.org/abs/2102.12474) with
               training-inference decoupling for UAV deployment scenarios.
    """

    def __init__(self, epsilon=1e-4, init_gamma=0.0):
        """
        Initialize the ERAM module

        Parameters:
        -----------
        epsilon : float
            Numerical stabilization constant (ε in Eq. 3)
        init_gamma : float
            Initial value for learnable scaling factors (γ in Eq. 5)
            Set to 0 for stable convergence
        """
        super(ERAM, self).__init__()
        self.epsilon = epsilon
        # Learnable scaling factors γ ∈ ℝ^C (initialized to 0)
        self.register_parameter('gamma', None)
        self.init_gamma = init_gamma

    def forward(self, x):
        """
        Forward pass implementing energy-based attention

        Parameters:
        -----------
        x : torch.Tensor
            Input feature tensor x(c,i,j) of shape [N, C, H, W]

        Returns:
        --------
        torch.Tensor
            During training: x̂(c,i,j) = γ_c · x(c,i,j) ⊕ (1-γ_c) · x(c,i,j) ⊙ Att(c,i,j)
            During inference: x̂(c,i,j) = x(c,i,j) (identity)
        """
        N, C, H, W = x.shape

        # Initialize gamma on first forward pass
        if self.gamma is None:
            self.gamma = nn.Parameter(torch.full((C,), self.init_gamma,
                                                 device=x.device, dtype=x.dtype))

        if self.training:
            mu_c = x.mean(dim=[2, 3], keepdim=True)  # μ_c
            var_c = ((x - mu_c) ** 2).mean(dim=[2, 3], keepdim=True)  # σ_c^2
            lambda_val = 1.0 / (H * W - 1)
            att = (x - mu_c) ** 2 / (lambda_val * var_c + self.epsilon)

            # Normalize attention to [0, 1] range
            att = att / (att.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + self.epsilon)

            # Reshape gamma for broadcasting
            gamma = self.gamma.view(1, C, 1, 1)
            out = gamma * x + (1 - gamma) * x * att

            return out
        else:

            return x