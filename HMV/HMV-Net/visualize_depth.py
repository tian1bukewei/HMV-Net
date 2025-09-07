import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import matplotlib


def visualize_depth_comparison(rgb_path, danv2_depth_path, hmvnet_depth_path, save_path=None):
    """
    Create a visualization comparing DAN-v2 and HMV-Net depth maps

    This visualization demonstrates HMV-Net's advantages in foreground accuracy
    and boundary preservation for UAV cargo delivery scenarios.

    Parameters:
    -----------
    rgb_path : str
        Path to input RGB image
    danv2_depth_path : str
        Path to DAN-v2 baseline depth map
    hmvnet_depth_path : str
        Path to HMV-Net depth map (with ERAM and SEU enhancement)
    save_path : str, optional
        Path to save the visualization
    """
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Column titles
    col_titles = ["Input RGB", "DAN-v2", "HMV-Net (Ours)"]

    # Load images
    rgb_img = cv2.imread(rgb_path)
    if rgb_img is None:
        raise ValueError(f"Could not load RGB image: {rgb_path}")
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    # Load depth maps
    danv2_depth = cv2.imread(danv2_depth_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if danv2_depth is None:
        danv2_depth = cv2.imread(danv2_depth_path, cv2.IMREAD_GRAYSCALE)

    hmvnet_depth = cv2.imread(hmvnet_depth_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if hmvnet_depth is None:
        hmvnet_depth = cv2.imread(hmvnet_depth_path, cv2.IMREAD_GRAYSCALE)

    # Normalize depth maps for visualization
    def normalize_depth(depth):
        if depth.max() > depth.min():
            return (depth - depth.min()) / (depth.max() - depth.min())
        return depth

    danv2_norm = normalize_depth(danv2_depth)
    hmvnet_norm = normalize_depth(hmvnet_depth)

    # Use Spectral_r colormap for depth visualization (consistent with DAN-v2)
    depth_cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    # Display images
    axes[0].imshow(rgb_img)
    axes[0].set_title(col_titles[0], fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(danv2_norm, cmap=depth_cmap)
    axes[1].set_title(col_titles[1], fontsize=14, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(hmvnet_norm, cmap=depth_cmap)
    axes[2].set_title(col_titles[2], fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    plt.show()
    plt.close()


def visualize_structure_aware_comparison(rgb_path, danv2_depth_path, hmvnet_depth_path,
                                         save_path=None, show_difference=True):
    """
    Create structure-aware visualization showing foreground improvements

    Parameters:
    -----------
    rgb_path : str
        Path to input RGB image
    danv2_depth_path : str
        Path to DAN-v2 depth map
    hmvnet_depth_path : str
        Path to HMV-Net depth map
    save_path : str, optional
        Path to save the visualization
    show_difference : bool
        Whether to show difference map
    """
    # Create figure with 4 subplots if showing difference
    ncols = 4 if show_difference else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    # Load images
    rgb_img = cv2.imread(rgb_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    danv2_depth = cv2.imread(danv2_depth_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if danv2_depth is None:
        danv2_depth = cv2.imread(danv2_depth_path, cv2.IMREAD_GRAYSCALE)

    hmvnet_depth = cv2.imread(hmvnet_depth_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if hmvnet_depth is None:
        hmvnet_depth = cv2.imread(hmvnet_depth_path, cv2.IMREAD_GRAYSCALE)

    # Convert to float for processing
    danv2_depth = danv2_depth.astype(np.float32)
    hmvnet_depth = hmvnet_depth.astype(np.float32)

    # Normalize for visualization
    def normalize_depth(depth):
        return (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    danv2_norm = normalize_depth(danv2_depth)
    hmvnet_norm = normalize_depth(hmvnet_depth)

    # Colormap
    depth_cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    # Display RGB
    axes[0].imshow(rgb_img)
    axes[0].set_title("Input RGB", fontsize=12)
    axes[0].axis('off')

    # Display DAN-v2
    axes[1].imshow(danv2_norm, cmap=depth_cmap)
    axes[1].set_title("DAN-v2", fontsize=12)
    axes[1].axis('off')

    # Display HMV-Net
    axes[2].imshow(hmvnet_norm, cmap=depth_cmap)
    axes[2].set_title("HMV-Net (Ours)", fontsize=12)
    axes[2].axis('off')

    # Display difference map if requested
    if show_difference:
        # Compute difference (positive = HMV-Net closer, negative = DAN-v2 closer)
        diff = hmvnet_norm - danv2_norm

        # Use diverging colormap for difference
        diff_cmap = matplotlib.colormaps.get_cmap('RdBu_r')
        im = axes[3].imshow(diff, cmap=diff_cmap, vmin=-0.2, vmax=0.2)
        axes[3].set_title("Difference Map", fontsize=12)
        axes[3].axis('off')

        # Add colorbar
        plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.suptitle("Structure-Aware Depth Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Structure-aware visualization saved to: {save_path}")

    plt.show()
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize depth comparison between DAN-v2 and HMV-Net")
    parser.add_argument("--input", type=str, required=True, help="Path to input RGB image")
    parser.add_argument("--danv2", type=str, required=True, help="Path to DAN-v2 depth map")
    parser.add_argument("--hmvnet", type=str, required=True, help="Path to HMV-Net depth map")
    parser.add_argument("--output", type=str, default="comparison.png", help="Path to save visualization")
    parser.add_argument("--structure", action="store_true", help="Create structure-aware visualization")

    args = parser.parse_args()

    if args.structure:
        visualize_structure_aware_comparison(
            args.input,
            args.danv2,
            args.hmvnet,
            args.output.replace('.png', '_structure.png'),
            show_difference=True
        )
    else:
        visualize_depth_comparison(
            args.input,
            args.danv2,
            args.hmvnet,
            args.output
        )