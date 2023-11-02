import cv2
import numpy as np
from ultralytics import YOLO

# Inicializa o modelo YOLO
yolo_model = YOLO('best.pt')
letter_model = YOLO('letter.pt')

# Abrir a captura de vídeo
camera_ids = [0]  # IDs das câmeras
captures = [cv2.VideoCapture(id) for id in camera_ids]

yolo_classes = {0: "Blue Box", 1: "Green Box"}
letter_classes = {0: "A", 1: "B", 2: "C"}

while all([cap.isOpened() for cap in captures]):
    frames = [cap.read()[1] for cap in captures]
    results = [yolo_model(frame) for frame in frames]

    for frame, res in zip(frames, results):
        boxes = res[0].boxes
        confidences = boxes.conf  # Get confidence scores
        classes = boxes.cls  # Get class indices
        xyxys = boxes.xyxy

        # Iterate over all detected boxes
        if len(confidences) > 0:
            if max(confidences) > 0.75:
                higher_conf = np.argmax(confidences)
                box = xyxys[higher_conf]
                conf = confidences[higher_conf]
                cls_idx = classes[higher_conf]

                # Box coordinates
                x1, y1, x2, y2 = map(int, box[:4])

                # Extract ROI from the frame
                roi = frame[y1:y2, x1:x2]
                
                result = letter_model(roi)

                # Annotate original frame with YOLO box, class, and confidence
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                yolo_label = f"{yolo_classes[int(cls_idx)]}: {conf:.2f}" 
                letter_label = f"Letter {letter_classes[result[0].probs.top1]}: {result[0].probs.top1conf:.2f}"
                cv2.putText(frame, yolo_label, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, letter_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Resize frames for concatenation
        original_frame_resized = cv2.resize(frame, (640, 480))

        # Display the concatenated images
        cv2.imshow("Inference", original_frame_resized)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Release all captures and destroy all windows after the loop is exited
for cap in captures:
    cap.release()
cv2.destroyAllWindows()
