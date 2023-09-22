import cv2
import numpy as np
from ultralytics import YOLO

def detect_cameras(max_cameras=10):
    """Detecta as câmeras conectadas ao sistema e retorna seus índices."""
    connected_cameras = []
    
    # Tenta inicializar VideoCapture para vários índices
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # A câmera está conectada
            connected_cameras.append(i)
            cap.release()
    
    return connected_cameras

# Detecta as câmeras conectadas
camera_ids = detect_cameras()
captures = [cv2.VideoCapture(id) for id in camera_ids]

# Inicializa o modelo YOLO
model = YOLO('best.pt')

while all([cap.isOpened() for cap in captures]):
    frames = [cap.read()[1] for cap in captures]
    results = [model(frame) for frame in frames]
    annotated_frames = [res[0].plot() for res in results]

    # Combina as imagens lado a lado
    combined_frame = np.hstack(annotated_frames)
    
    cv2.imshow("YOLOv8 Inference", combined_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libera os objetos VideoCapture e fecha todas as janelas
for cap in captures:
    cap.release()
cv2.destroyAllWindows()
