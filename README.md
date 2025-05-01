# TrackGuard TensorRT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorRT](https://img.shields.io/badge/TensorRT-8.0%2B-green.svg)](https://developer.nvidia.com/tensorrt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

TrackGuard TensorRT is a high-performance object tracking system optimized with TensorRT for real-time inference on NVIDIA GPUs including Jetson devices. It combines YOLOv8 detection with ByteTrack-based tracking algorithm for efficient multi-object tracking.

<img src="https://github.com/yourusername/trackguard-tensorrt/assets/images/preview.png" width="800" alt="TrackGuard TensorRT demo">

## 🚀 Features

- **TensorRT Optimization**: Up to 3x faster inference compared to standard PyTorch
- **YOLOv8 Support**: Compatible with YOLOv8n/s/m/l/x models and custom trained models
- **Jetson Compatibility**: Optimized for edge deployment on NVIDIA Jetson devices
- **MOT Evaluation**: Support for MOT Challenge metrics and evaluation
- **Custom Tracking**: Advanced tracking with configurable parameters
- **Real-time Performance**: Achieve 60+ FPS on consumer GPUs

## 📋 Requirements

### Core Dependencies

- Python 3.8 or higher
- CUDA 11.4+ (recommended: CUDA 11.8 or 12.0)
- TensorRT 8.0+
- PyTorch 2.0+
- Ultralytics 8.0.0+
- OpenCV
- PyCUDA

### Additional Libraries

- ONNX and ONNX Runtime
- Matplotlib, Pandas, tqdm (for evaluation and visualization)

## 🔧 Installation

### Option 1: Using pip (Recommended)

```bash
# Basic libraries
pip install ultralytics opencv-python numpy matplotlib pandas tqdm

# ONNX support
pip install onnx onnxruntime-gpu

# TensorRT and CUDA support
pip install nvidia-pyindex
pip install nvidia-tensorrt
pip install pycuda

# Clone repository
git clone https://github.com/yourusername/trackguard-tensorrt.git
cd trackguard-tensorrt
```

### Option 2: Using conda

```bash
# Create conda environment
conda create -n trackguard python=3.9
conda activate trackguard

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install ultralytics opencv-python matplotlib pandas tqdm
pip install onnx onnxruntime-gpu
pip install nvidia-pyindex nvidia-tensorrt pycuda

# Clone repository
git clone https://github.com/yourusername/trackguard-tensorrt.git
cd trackguard-tensorrt
```

### For Jetson Devices

```bash
# TensorRT is already included in JetPack SDK

# Install dependencies
sudo apt-get update
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev
pip3 install ultralytics opencv-python matplotlib pandas tqdm
pip3 install pycuda onnx onnxruntime

# Clone repository
git clone https://github.com/yourusername/trackguard-tensorrt.git
cd trackguard-tensorrt
```

## 🔍 Verify Installation

Run the verification script to ensure all components are installed correctly:

```bash
python verify_tensorrt.py
```

## 🛠️ Usage

### Step 1: Convert YOLOv8 Model to TensorRT

```bash
# Direct conversion using Ultralytics
yolo export model=yolov8n.pt format=engine half=True device=0

# Or manual two-step conversion
yolo export model=yolov8n.pt format=onnx
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.engine --fp16
```

### Step 2: Run Inference

```bash
# Process video with TensorRT engine
python main_tensorrt.py --input video.mp4 --engine yolov8n.engine --output output.mp4

# For real-time webcam inference
python main_tensorrt.py --input 0 --engine yolov8n.engine
```

### Step 3: Evaluate Performance

```bash
# Evaluate on MOT Challenge dataset
python main_evaluation_tensorrt.py --dataset path/to/mot/dataset --engine yolov8n.engine

# Compare with PyTorch performance
python run_mot_evaluation_tensorrt.py --dataset path/to/mot/dataset --engine yolov8n.engine --compare
```

## 📊 Performance

| Model | Format | MOTA | MOTP | IDF1 | Precision | Recall | FPS (RTX 3060) | FPS (Jetson) |
|-------|--------|------|------|------|-----------|--------|----------------|--------------|
| YOLOv8n | PyTorch | 0.457 | 0.857 | 0.646 | 0.930 | 0.494 | 48.3 | 12.5 |
| YOLOv8n | TensorRT | 0.457 | 0.857 | 0.646 | 0.930 | 0.494 | 76.0 | 19.8 |
| Custom Fine-Tuned | PyTorch | 0.499 | 0.912 | 0.674 | 0.964 | 0.518 | 44.3 | 10.2 |
| Custom Fine-Tuned | TensorRT | 0.498 | 0.912 | 0.674 | 0.963 | 0.518 | 65.1 | 16.4 |

### Speedup Analysis

| Metric | PyTorch | TensorRT | Speedup |
|--------|---------|----------|---------|
| Inference Time | 20.7 ms | 13.15 ms | 1.57x |
| Total Processing | 22.6 ms | 15.35 ms | 1.47x |
| End-to-End FPS | 28.0 | 31.1 | 1.11x |

## 📁 Project Structure

```
trackguard-tensorrt/
├── src/
│   ├── core/
│   │   ├── track_manager.py         # Core tracking algorithms
│   │   ├── bbox_handler.py          # Bounding box utilities
│   │   └── confidence_handler.py    # Confidence management
│   ├── models/
│   │   ├── yolo_detector.py         # PyTorch YOLOv8 detector
│   │   └── yolo_tensorrt_detector.py # TensorRT optimized detector
│   ├── analyzers/
│   │   ├── color_analyzer.py        # Color appearance model
│   │   └── shape_analyzer.py        # Shape appearance model
│   ├── evaluation/
│   │   ├── mot_metrics.py           # MOT Challenge metrics
│   │   └── visualizer.py            # Performance visualization
│   └── utils/
│       ├── tensorrt_utils.py        # TensorRT conversion utilities
│       └── mot_utils.py             # MOT evaluation utilities
├── main.py                          # Standard tracking with PyTorch
├── main_tensorrt.py                 # TensorRT-optimized tracking
├── main_evaluation.py               # MOT evaluation with PyTorch
├── main_evaluation_tensorrt.py      # MOT evaluation with TensorRT
├── convert_to_tensorrt.py           # Model conversion utility
├── verify_tensorrt.py               # Installation verification
└── README.md                        # This file
```

## 📊 Results Visualization

<img src="https://github.com/yourusername/trackguard-tensorrt/assets/images/performance_chart.png" width="600" alt="Performance Chart">

## ⚙️ Configuration

TrackGuard TensorRT supports extensive configuration options:

```json
{
  "min_confidence": 0.6,
  "high_det_thresh": 0.7,
  "low_det_thresh": 0.1,
  "match_thresh": 0.8,
  "max_age": 15,
  "min_hits": 3,
  "base_iou_threshold": 0.3,
  "edge_conf": 0.65,
  "small_edge_conf": 0.7
}
```

Pass a JSON configuration file using the `--config-file` parameter.

## 🔍 Advanced Usage

### Custom Training and Fine-Tuning

```bash
# Train YOLOv8 on custom dataset
yolo train model=yolov8n.pt data=custom.yaml epochs=100 imgsz=640

# Export the fine-tuned model to TensorRT
yolo export model=runs/detect/train/weights/best.pt format=engine
```

### Running on Jetson

```bash
# Make sure to export model on Jetson itself
yolo export model=yolov8n.pt format=engine half=True device=0 workspace=4

# Run inference with resource monitoring
python main_tensorrt.py --input video.mp4 --engine yolov8n.engine --monitor-resources
```

### Batch Processing

```bash
# Process multiple videos
python batch_process.py --input_dir /path/to/videos --engine yolov8n.engine --output_dir /path/to/outputs
```

## 📝 Citation

If you use TrackGuard TensorRT in your research, please cite our paper:

```bibtex
@article{yourname2025trackguard,
  title={TrackGuard TensorRT: Efficient Multi-Object Tracking for Edge Devices},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={2025},
  volume={X},
  pages={XXX-XXX}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- [MOT Challenge](https://motchallenge.net/)

## 📧 Contact

Your Name - [@your_twitter](https://twitter.com/your_twitter) - email@example.com

Project Link: [https://github.com/yourusername/trackguard-tensorrt](https://github.com/yourusername/trackguard-tensorrt)
