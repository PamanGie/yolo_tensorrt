import cv2
from src.models.yolo_tensorrt_detector import YOLOTensorRTDetector

# Inisialisasi detector TensorRT
detector = YOLOTensorRTDetector(engine_path="yolov8x.engine", conf_threshold=0.25)

# Buka video
video_path = "path/to/video.mp4"
cap = cv2.VideoCapture(video_path)

# Cek apakah video berhasil dibuka
if not cap.isOpened():
    print(f"Error: Tidak dapat membuka video {video_path}")
    exit()

# Dapatkan informasi video
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Setup video writer jika ingin menyimpan hasil
output_path = "output.mp4"
writer = cv2.VideoWriter(output_path, 
                        cv2.VideoWriter_fourcc(*'mp4v'), 
                        fps, 
                        (width, height))

# Proses video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Lakukan deteksi pada frame
    detections = detector.detect(frame)
    
    # Visualisasi hasil deteksi
    for det in detections:
        bbox = det['bbox']
        conf = det['confidence']
        class_id = det['class_id']
        
        # Dapatkan nama kelas (jika tersedia di detector)
        class_name = detector.class_names[class_id] if hasattr(detector, 'class_names') else f"Class {class_id}"
        
        # Gambar bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Tambahkan label
        label = f"{class_name} {conf:.2f}"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Tampilkan frame
    cv2.imshow('Video', frame)
    
    # Simpan frame ke output video
    writer.write(frame)
    
    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan resources
cap.release()
writer.release()
cv2.destroyAllWindows()
detector.cleanup()  # Bersihkan resources TensorRT

print(f"Video berhasil diproses dan disimpan ke {output_path}")
