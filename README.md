#Method
Our method enhances monocular depth estimation through two key innovations:
1.Energy-based Residual Attention Module(ERAM): Implements spatially-adaptive feature modulation during training through energy minimization principles while reducing to identity mapping during inference.
  During training: Applies energy-based spatial attention to identify and enhance foreground-critical regions through residual connections. 
  At inference: Reduces to identity transformation, ensuring zero computational overhead.
2.Semantic Enhancement Unit（SEU): A complementary component that leverages ERAM's attention results to perform adaptive enhancement on focused regions.
This module implements lightweight post-processing on attention-identified critical regions through depth-guided residual correction.
The combination of these techniques results in more detailed and perceptually accurate depth maps, particularly for foreground objects and complex scenes.

Code Structure
.
├── ERAM.py      # Implementation of Self-calibrated Attention Module
├── SEU.py       # Enhanced Depth Perception Transformer with foreground boosting
├── visualize_depth.py    # Visualization and comparison utilities
└── requirements.txt      # Dependencies

Uasge Requirements
python>=3.10.0
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.3
numpy>=1.20.0
matplotlib>=3.4.0

Model Weights
DAN-v2(-S)，You can download it from the official repository of DAN-v2.

Running the Model
  # Load the enhanced model
  from enhanced_dpt import EnhancedDepthAnythingV2
  
  # Initialize with foreground enhancement
  model = EnhancedDepthAnythingV2(
      encoder='vit',              # Choose from: 'vits', 'vitb', 'vitl', 'vitg'
      enable_NoEffectSimAm=True,           # Enable Self-calibrated Attention
      foreground_boost=0.5,        # Foreground enhancement strength
      foreground_threshold=0.3     # Threshold for foreground/background separation
  )
  
  # Load weights
  model.load_state_dict(torch.load('path/to/depth_anything_v2_vits.pth'), strict=False)
  model = model.to(device).eval()
  
  # Run inference
  with torch.no_grad():
      depth_map = model.infer_image(image)

visualization Tool
We provide a visualization tool to compare our HMV-Net model with the original Depth Anything V2:
python visualize_depth.py --img-path path/to/image.jpg --output-dir results --checkpoint-path path/to/checkpoints --enhanced-viz
