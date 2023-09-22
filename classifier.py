import torch
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('best.pt')
cap = cv2.VideoCapture(0)

# For calculating FPS
font = cv2.FONT_HERSHEY_SIMPLEX
color = (0, 255, 0)  # Green
font_scale = 0.7
thickness = 2
prev_frame_time = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    new_frame_time = cv2.getTickCount()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Calculate FPS
        fps = cv2.getTickFrequency() / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps_text = f"FPS: {fps:.2f}"

        # Put the FPS on the top-left corner of the frame
        cv2.putText(annotated_frame, fps_text, (10, 30), font, font_scale, color, thickness)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
