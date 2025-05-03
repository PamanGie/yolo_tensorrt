from ultralytics import YOLO
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

def run_inference(model_path, is_tensorrt=False, num_frames=None):
    # Load model
    if is_tensorrt:
        model = YOLO(model_path)  # TensorRT model (.engine)
    else:
        model = YOLO(model_path)  # Standard model (.pt)
    
    # Untuk video
    video_path = "video.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if num_frames is None:
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Variabel untuk tracking FPS
    fps_list = []
    frame_count = 0
    
    # Loop melalui frame
    start_time_total = time.time()
    while cap.isOpened() and frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Inference dengan penanganan error
        try:
            start_time = time.time()
            results = model(frame)
            inference_time = time.time() - start_time
            
            # Hindari pembagian oleh nol
            if inference_time > 0:
                fps = 1.0 / inference_time
            else:
                # Gunakan nilai maksimum yang wajar jika terlalu cepat untuk diukur
                fps = 1000.0  # 1000 FPS sebagai batas atas yang wajar
                print(f"Warning: Frame {frame_count} - Inference time terlalu cepat untuk diukur")
            
            fps_list.append(fps)
            
            # Update frame count
            frame_count += 1
            
            # Hitung average FPS sejauh ini (hindari list kosong)
            if fps_list:
                avg_fps = np.mean(fps_list)
                
                # Print progress
                if frame_count % 10 == 0:
                    print(f"Processing frame {frame_count}/{num_frames}, Current FPS: {fps:.1f}, Avg FPS: {avg_fps:.1f}")
        except Exception as e:
            print(f"Error pada frame {frame_count}: {str(e)}")
            # Lanjutkan ke frame berikutnya jika terjadi error
            frame_count += 1
            continue
    
    # Hitung total waktu dan average FPS
    total_time = time.time() - start_time_total
    # Hindari pembagian oleh nol
    if total_time > 0 and frame_count > 0:
        final_avg_fps = frame_count / total_time
    else:
        final_avg_fps = 0
    
    cap.release()
    
    return {
        "fps_list": fps_list,
        "avg_fps": final_avg_fps,
        "total_time": total_time,
        "frames": frame_count
    }

def compare_models(standard_model, tensorrt_model, num_frames=100):
    print(f"Running comparison on {num_frames} frames...")
    
    # Run standard model
    print("\nTesting Standard YOLO model...")
    std_results = run_inference(standard_model, is_tensorrt=False, num_frames=num_frames)
    
    # Run TensorRT model
    print("\nTesting TensorRT model...")
    trt_results = run_inference(tensorrt_model, is_tensorrt=True, num_frames=num_frames)
    
    # Output comparison
    print("\n=== PERFORMANCE COMPARISON ===")
    print(f"Standard YOLO: {std_results['avg_fps']:.2f} FPS (avg)")
    print(f"TensorRT YOLO: {trt_results['avg_fps']:.2f} FPS (avg)")
    
    # Hindari pembagian oleh nol
    if std_results['avg_fps'] > 0:
        speedup = trt_results['avg_fps'] / std_results['avg_fps']
        print(f"Speed Improvement: {speedup:.2f}x faster")
    else:
        print("Speed Improvement: Cannot calculate (standard model FPS too low or zero)")
    
    # Buat grafik hanya jika kedua model berhasil menghasilkan data
    if len(std_results['fps_list']) > 0 and len(trt_results['fps_list']) > 0:
        # Pastikan panjang data sama untuk plot yang adil
        min_length = min(len(std_results['fps_list']), len(trt_results['fps_list']))
        std_fps_list = std_results['fps_list'][:min_length]
        trt_fps_list = trt_results['fps_list'][:min_length]
        
        # Create comparison graph
        plt.figure(figsize=(12, 6))
        
        # Plot FPS over time
        plt.subplot(1, 2, 1)
        plt.plot(std_fps_list, label='Standard YOLO')
        plt.plot(trt_fps_list, label='TensorRT')
        plt.title('FPS per Frame')
        plt.xlabel('Frame')
        plt.ylabel('FPS')
        plt.legend()
        plt.grid(True)
        
        # Plot average FPS comparison
        plt.subplot(1, 2, 2)
        bars = plt.bar(['Standard YOLO', 'TensorRT'], 
                [std_results['avg_fps'], trt_results['avg_fps']])
        plt.title('Average FPS Comparison')
        plt.ylabel('FPS')
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('yolo_tensorrt_comparison.png')
        plt.show()
    else:
        print("Cannot generate comparison graph: Insufficient data from one or both models")
    
    return std_results, trt_results

if __name__ == "__main__":
    # Define paths to models
    standard_model = "yolov8n.pt"     # Standard YOLO model
    tensorrt_model = "yolov8n.engine" # TensorRT model
    
    try:
        # Run comparison (test on 100 frames for quicker results)
        std_results, trt_results = compare_models(standard_model, tensorrt_model, num_frames=100)
    except Exception as e:
        print(f"Error during comparison: {str(e)}")