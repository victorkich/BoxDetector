import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ultralytics import YOLO


# Basic Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Adjusting dimensions if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


# ResNet-32 Model
class ResNet32(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet32, self).__init__()
        self.in_channels = 16  # Assuming the same initial number of channels as LeNet-3

        # Initial convolution
        self.conv = nn.Conv2d(1, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        # Stacking residual blocks
        self.layer1 = self.make_layer(ResidualBlock, 16, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 32, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 64, 2, stride=2)

        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels  # Update in_channels for the next iteration
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the model, loss function, and optimizer
model = ResNet32().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# model = torch.load('resnet.pt')
model.load_state_dict(torch.load('./weights/resnet.pth', map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
model.eval()

# Inicializa o modelo YOLO
yolo_model = YOLO('best.pt')

# Transforms para a imagem antes de alimentar o modelo LeNet
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((320, 320)),
    transforms.ToTensor()
])

# Define the zoom factor
zoom_factor = 2 # Adjust the zoom factor as needed

# Abrir a captura de vídeo
camera_ids = [0]  # IDs das câmeras
captures = [cv2.VideoCapture(id) for id in camera_ids]

yolo_classes = {0: "Blue Box", 1: "Green Box"}
letter_classes = {0: "A", 1: "B", 2: "C"}
first_time = True


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
            if max(confidences) > 0.80:
                higher_conf = np.argmax(confidences)
                box = xyxys[higher_conf]
                conf = confidences[higher_conf]
                cls_idx = classes[higher_conf]

                # Box coordinates
                x1, y1, x2, y2 = map(int, box[:4])
                # Extract ROI from the frame
                roi = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                # Prepare the image for LeNet classification
                thresholded_in_pil = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
                input_image = transform(thresholded_in_pil)
                input_image = input_image.unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

                # Classify with LeNet
                with torch.no_grad():
                    output = model(input_image)
                _, predicted = torch.max(output, 1)
                predicted_class = predicted.item()
                predicted_conf = torch.softmax(output, dim=1)[0][predicted.item()].item()

                # Annotate original frame with YOLO box, class, and confidence
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                yolo_label = f"{yolo_classes[int(cls_idx)]}: {conf:.2f}" 
                cv2.putText(frame, yolo_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Annotate ROI with LeNet predicted class and confidence
                lenet_label = f"Letter: {letter_classes[predicted_class]} {predicted_conf:.2f}"
                cv2.putText(gray, lenet_label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Resize frames for concatenation
                original_frame_resized = cv2.resize(frame, (640, 480))
                gray_resized_bgr = cv2.cvtColor(cv2.resize(gray, (640, 480)), cv2.COLOR_GRAY2BGR)

                # Concatenate the resized frames horizontally
                concatenated_images = cv2.hconcat([original_frame_resized, gray_resized_bgr])
            else:
                # Resize frames for concatenation
                original_frame_resized = cv2.resize(frame, (640, 480))
                gray = cv2.cvtColor(original_frame_resized, cv2.COLOR_BGR2GRAY)
                gray_resized_bgr = cv2.cvtColor(cv2.resize(gray, (640, 480)), cv2.COLOR_GRAY2BGR)

                # Concatenate the resized frames horizontally
                concatenated_images = cv2.hconcat([original_frame_resized, gray_resized_bgr])
        else:
            # Resize frames for concatenation
            original_frame_resized = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(original_frame_resized, cv2.COLOR_BGR2GRAY)
            gray_resized_bgr = cv2.cvtColor(cv2.resize(gray, (640, 480)), cv2.COLOR_GRAY2BGR)

            # Concatenate the resized frames horizontally
            concatenated_images = cv2.hconcat([original_frame_resized, gray_resized_bgr])

        # Display the concatenated images
        cv2.imshow("Inference", concatenated_images)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Release all captures and destroy all windows after the loop is exited
for cap in captures:
    cap.release()
cv2.destroyAllWindows()
