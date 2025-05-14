import torch
import torch.nn as nn


class NoEffectSimAM(nn.Module):
    """
    No-Effect Simple Attention Module (NoEffectSimAM): Introduces attention-driven
    regularization into the training process without altering inference-time behavior.

    During training: Injects structured gradients that encourage spatial discrimination
                    and boundary awareness via energy-based attention.
    At inference: Reverts to an identity function, ensuring zero computational overhead.

    Based on the modified version of "SimAM: A Simple, Parameter-Free Attention Module
    for Convolutional Neural Networks" (https://arxiv.org/abs/2102.12474)
    """

    def __init__(self, e_lambda=1e-5, alpha=1e-6):
        """
        Initialize the NoEffectSimAM module

        Parameters:
        -----------
        e_lambda : float
            Small constant for numerical stability in attention computation (ε in Eq. 3)
        alpha : float
            Modulation scalar that controls the strength of attention (α_c in Eq. 4),
            initialized to a small value (e.g., 1e-6)
        """
        super(NoEffectSimAM, self).__init__()
        self.e_lambda = e_lambda
        # Learnable modulation scalar (Eq. 4)
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))

    def forward(self, x):
        """
        Forward pass computing spatial attention via energy normalization
        and applying residual blending.

        Parameters:
        -----------
        x : torch.Tensor
            Input feature tensor of shape [N, C, H, W]

        Returns:
        --------
        torch.Tensor
            Output tensor with attention applied during training,
            or original input during inference
        """
        if self.training:
            # Channel-wise mean (Eq. 1)
            mu = x.mean(dim=[2, 3], keepdim=True)

            # Spatial deviation (Eq. 2)
            delta = (x - mu).pow(2)

            # Variance term for energy normalization
            energy = delta.mean(dim=[2, 3], keepdim=True)

            # Attention map computation (Eq. 3)
            att = (4 * energy + self.e_lambda) / (delta + self.e_lambda)

            # Residual modulation (Eq. 4)
            out = (1 - self.alpha) * x + self.alpha * x * att

            return out
        else:
            # During inference, revert to identity (α_c = 0)
            return x