#Method
Our method enhances monocular depth estimation through two key innovations:
1.Energy-based Residual Attention Module(ERAM): We integrate a parameter-free attention mechanism that adaptively refines feature representations by calculating channel and spatial attention based on statistical measures of feature activation. This module enhances the network's focus on salient regions without introducing additional trainable parameters or computational complexity.
2.Semantic Enhancement Unit（SEU): We employ a depth-aware foreground enhancement technique that dynamically adjusts contrast for objects closer to the camera. This process involves:
  Creating a foreground probability map from normalized depth values
  Applying an adaptive enhancement factor proportional to foreground probability
  Preserving relative depth relationships while improving foreground-background separation
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
We provide a visualization tool to compare our RASE model with the original Depth Anything V2:
python visualize_depth.py --img-path path/to/image.jpg --output-dir results --checkpoint-path path/to/checkpoints --enhanced-viz
