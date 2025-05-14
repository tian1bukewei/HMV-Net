import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import matplotlib


def visualize_depth_comparison(rgb_path, danv2_depth_path, rase_depth_path, save_path=None):
    """
    Create a visualization comparing DAN-v2 and RASE-Net depth maps

    Parameters:
    -----------
    rgb_path : str
        Path to input RGB image
    danv2_depth_path : str
        Path to DAN-v2 depth map
    rase_depth_path : str
        Path to RASE-Net depth map (with SimAM and foreground boosting)
    save_path : str, optional
        Path to save the visualization
    """
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Column titles
    col_titles = ["Input Image", "DAN-v2 Depth", "RASE-Net Depth"]

    # Load images
    rgb_img = cv2.imread(rgb_path)
    if rgb_img is None:
        raise ValueError(f"Could not load RGB image: {rgb_path}")
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    danv2_depth = cv2.imread(danv2_depth_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if danv2_depth is None:
        danv2_depth = cv2.imread(danv2_depth_path, cv2.IMREAD_GRAYSCALE)

    rase_depth = cv2.imread(rase_depth_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if rase_depth is None:
        rase_depth = cv2.imread(rase_depth_path, cv2.IMREAD_GRAYSCALE)

    # Normalize depth maps
    def normalize_depth(depth):
        if depth.max() > depth.min():
            return (depth - depth.min()) / (depth.max() - depth.min())
        return depth

    danv2_norm = normalize_depth(danv2_depth)
    rase_norm = normalize_depth(rase_depth)

    # Get the Spectral_r colormap used by Depth Anything V2
    depth_cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    # Display images
    axes[0].imshow(rgb_img)
    axes[0].set_title(col_titles[0], fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(danv2_norm, cmap=depth_cmap)
    axes[1].set_title(col_titles[1], fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(rase_norm, cmap=depth_cmap)
    axes[2].set_title(col_titles[2], fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize depth comparison between DAN-v2 and RASE-Net")
    parser.add_argument("--input", type=str, help="Path to input RGB image")
    parser.add_argument("--danv2", type=str, help="Path to DAN-v2 depth map")
    parser.add_argument("--rase", type=str, help="Path to RASE-Net depth map")
    parser.add_argument("--output", type=str, default="comparison.png", help="Path to save visualization")

    args = parser.parse_args()

    visualize_depth_comparison(
        args.input,
        args.danv2,
        args.rase,
        args.output
    )