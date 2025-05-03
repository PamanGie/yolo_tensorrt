from ultralytics import YOLO
import cv2
import time
import numpy as np

# Load model TensorRT
model = YOLO("yolov8n.engine")

# Untuk video
video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

# Variabel untuk tracking FPS
fps_list = []
frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
start_time_total = time.time()

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
    fps_list.append(fps)
    
    # Update frame count
    frame_count += 1
    
    # Hitung average FPS sejauh ini
    avg_fps = np.mean(fps_list)
    
    # Visualisasi
    annotated_frame = results[0].plot()
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Avg FPS: {avg_fps:.1f}", (20, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", (20, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Tampilkan
    cv2.imshow("YOLOv8 TensorRT", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Hitung total waktu dan average FPS
total_time = time.time() - start_time_total
final_avg_fps = frame_count / total_time

print(f"\n--- TensorRT Performance ---")
print(f"Total frames processed: {frame_count}")
print(f"Total time: {total_time:.2f} seconds")
print(f"Average FPS: {final_avg_fps:.2f}")
print(f"Average inference time per frame: {1000/final_avg_fps:.2f} ms")

cap.release()
cv2.destroyAllWindows()