# YOLOv8 dengan TensorRT - Optimasi Inference

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorRT](https://img.shields.io/badge/TensorRT-8.0%2B-green.svg)](https://developer.nvidia.com/tensorrt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

Tutorial dan implementasi untuk mengoptimalkan model YOLOv8 dengan TensorRT untuk inference yang lebih cepat. Repositori ini menyertakan kode untuk tutorial YouTube tentang cara meningkatkan kecepatan inference model YOLOv8 menggunakan TensorRT.

## 🚀 Keuntungan TensorRT

- 🔥 **Peningkatan Kecepatan**: Hingga 3x lebih cepat dibandingkan PyTorch standar
- 💾 **Efisiensi Memori**: Penggunaan memori yang lebih efisien
- 🔋 **Konsumsi Daya Lebih Rendah**: Ideal untuk edge devices seperti NVIDIA Jetson
- 🎯 **Presisi Fleksibel**: Dukungan untuk FP32, FP16, dan bahkan INT8

## 📋 Library yang Dibutuhkan

### Library Utama

```
torch
torchvision
torch-tensorrt
ultralytics
opencv-python
```

### Library Tambahan (untuk visualisasi dan demo)

```
numpy
matplotlib
tqdm
```

## 🔧 Instalasi

### Menggunakan pip

```bash
# Instal PyTorch dengan dukungan CUDA
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# Instal torch-tensorrt
pip install torch-tensorrt

# Instal Ultralytics YOLOv8
pip install ultralytics

# Instal library lainnya
pip install opencv-python numpy matplotlib tqdm
```

### Menggunakan Conda (Disarankan)

Conda menyediakan environment yang lebih terisolasi dan pengelolaan dependensi yang lebih baik, terutama untuk proyek yang melibatkan CUDA dan TensorRT.

```bash
# Buat environment conda baru
conda create -n yolo-tensorrt python=3.9
conda activate yolo-tensorrt

# Instal PyTorch dengan CUDA
-- Tidak Ada

# Instal torch-tensorrt
pip install torch-tensorrt

# Instal Ultralytics dan library lainnya
pip install ultralytics opencv-python matplotlib tqdm

# Verifikasi instalasi
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import torch_tensorrt; print(f'torch-tensorrt: {torch_tensorrt.__version__}')"
```

### Conda Environment dari File (Untuk Reproduksi)

Anda bisa menggunakan file environment.yml berikut untuk membuat environment yang identik:

```yaml
# environment.yml
name: yolo-tensorrt
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pip=22.3.1
  - pytorch=2.0.1
  - torchvision=0.15.1
  - pytorch-cuda=11.8
  - cudatoolkit=11.8
  - numpy=1.24.3
  - pip:
    - torch-tensorrt==1.4.0
    - ultralytics==8.0.200
    - opencv-python==4.8.0.76
    - matplotlib==3.7.2
    - tqdm==4.66.1
```

Gunakan perintah berikut untuk membuat environment dari file:

```bash
conda env create -f environment.yml
conda activate yolo-tensorrt
```

## ✅ Verifikasi Instalasi

Jalankan script berikut untuk memverifikasi bahwa TensorRT terinstal dengan benar:

```python
# verify_tensorrt.py
import torch
import torch_tensorrt

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")

print(f"torch-tensorrt version: {torch_tensorrt.__version__}")
```

## 🛠️ Penggunaan

### Konversi Model YOLOv8 ke TensorRT

#### Metode 1: Menggunakan API Ultralytics (Disarankan)

```bash
# Konversi langsung ke TensorRT engine
yolo export model=yolov8n.pt format=engine half=True device=0
```

#### Metode 2: Menggunakan torch-tensorrt

```python
# convert_yolo_tensorrt.py
import torch
import torch_tensorrt
from ultralytics import YOLO

# Load model YOLOv8
model = YOLO("yolov8n.pt")
pt_model = model.model.float().eval().cuda()

# Buat example input
example_input = torch.randn((1, 3, 640, 640)).cuda()

# Compile dengan TensorRT
trt_model = torch_tensorrt.compile(
    pt_model,
    ir="dynamo",
    inputs=[example_input],
    enabled_precisions=torch.float16,
    workspace_size=4 * 1024 * 1024 * 1024,
)

# Simpan model
torch_tensorrt.save(trt_model, "yolov8n_trt.ts")
```

### Inference dengan Model TensorRT

```python
# inference_tensorrt.py
from ultralytics import YOLO
import cv2
import time

# Load model TensorRT
model = YOLO("yolov8n.engine")

# Untuk video
video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

# Loop melalui frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Inference
    start_time = time.time()
    results = model(frame)
    inference_time = time.time() - start_time
    fps = 1.0 / inference_time
    
    # Visualisasi
    annotated_frame = results[0].plot()
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Tampilkan
    cv2.imshow("YOLOv8 TensorRT", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 📊 Pengukuran Performa

| Model | Format | Inference Time (ms) | FPS (RTX 3060) |
|-------|--------|---------------------|----------------|
| YOLOv8n | PyTorch | 20.7 | 48.3 |
| YOLOv8n | TensorRT | 13.2 | 76.0 |
| YOLOv8s | PyTorch | 30.5 | 32.8 |
| YOLOv8s | TensorRT | 19.8 | 50.5 |
| YOLOv8m | PyTorch | 48.7 | 20.5 |
| YOLOv8m | TensorRT | 29.3 | 34.1 |

Peningkatan kecepatan rata-rata: 1.5-1.7x

## 📝 Troubleshooting

### Masalah Umum dan Solusi

1. **Error: PendingUnbackedSymbolNotFound**:
   - Gunakan export via API Ultralytics (`yolo export`) alih-alih torch-tensorrt langsung

2. **CUDA out of memory**:
   - Kurangi ukuran input model (`imgsz`)
   - Gunakan model yang lebih kecil (YOLOv8n vs YOLOv8x)

3. **TensorRT not found**:
   - Pastikan CUDA dan cuDNN sudah terinstal dengan benar
   - Gunakan versi PyTorch yang kompatibel dengan versi CUDA Anda

4. **Model lambat di Jetson**:
   - Export model langsung di Jetson (jangan pindahkan engine dari PC ke Jetson)
   - Gunakan parameter `workspace=4` (atau lebih rendah) saat export di Jetson

5. **Masalah Conda environment**:
   - Jika terjadi konflik, buat environment baru dengan `conda clean -a` terlebih dahulu
   - Pastikan semua CUDA toolkit dan PyTorch menggunakan versi yang sama

### Troubleshooting Conda Khusus

**Masalah**: Conda tidak dapat menemukan paket torch-tensorrt  
**Solusi**: Instal dengan pip setelah menginstal PyTorch dengan Conda

**Masalah**: Konflik CUDA version  
**Solusi**:
```bash
# Periksa versi CUDA yang terpasang
nvcc --version

# Pastikan PyTorch menggunakan versi CUDA yang sama
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**Masalah**: Kesalahan "ModuleNotFoundError" untuk torch-tensorrt  
**Solusi**: Verifikasi instalasi dengan:
```bash
pip list | grep tensorrt
pip install --force-reinstall torch-tensorrt
```

## 🎬 Tutorial YouTube

Tonton tutorial lengkap di YouTube untuk pemahaman lebih dalam:
[YOLOv8 dengan TensorRT - Tutorial](https://youtube.com/your_channel)

## 📄 License

Project ini dilisensikan di bawah lisensi MIT - lihat file LICENSE untuk detail.

## 🙏 Referensi

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- [torch-tensorrt](https://github.com/pytorch/TensorRT)
