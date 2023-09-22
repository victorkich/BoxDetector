import cv2
import torch  # Added torch for the device selection
from ultralytics import YOLO

# Check if CUDA is available and use it, otherwise default to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = YOLO('best.pt').to(device)  # Move the model to the selected device
model = model.float()
cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Convert frame to torch tensor and move to device
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

        # Run YOLOv8 inference on the frame
        results = model(frame_tensor)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        annotated_frame = annotated_frame.permute(1, 2, 0).cpu().numpy()  # Convert tensor back to numpy ndarray for visualization

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
