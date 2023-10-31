import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ultralytics import YOLO

# Define the LeNet-3 architecture for A, B, C classification
class LeNet3(nn.Module):
    def __init__(self):
        super(LeNet3, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 3)  # Output for A, B, C

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
model = LeNet3().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

model.load_state_dict(torch.load('./weights/lenet.pth'))  # Replace with the actual path to your trained model
model.eval()

# Inicializa o modelo YOLO
yolo_model = YOLO('best.pt')

# Inicializa o modelo LeNet
lenet_model = LeNet3().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
lenet_model.load_state_dict(torch.load('./weights/lenet.pth'))
lenet_model.eval()

# Transforms para a imagem antes de alimentar o modelo LeNet
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# Abrir a captura de vídeo
camera_ids = [0]  # IDs das câmeras
captures = [cv2.VideoCapture(id) for id in camera_ids]

while all([cap.isOpened() for cap in captures]):
    frames = [cap.read()[1] for cap in captures]
    results = [yolo_model(frame) for frame in frames]
    
    for frame, res in zip(frames, results):
        annotated_frame = res[0].plot()
        
        boxes = res[0].boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            x1, y1, x2, y2 = b
            roi = frame[int(y1):int(y2), int(x1):int(x2)]  # Corta a região de interesse (bounding box)
            
            # Convert the image to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Apply thresholding to create a binary mask (white paper will be white, and the rest black)
            _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

            thresholded_in_pil = Image.fromarray(cv2.cvtColor(thresholded, cv2.COLOR_BGR2RGB))
            input_image = transform(thresholded_in_pil)
            input_image = input_image.unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            # Make a prediction using the LeNet model
            with torch.no_grad():
                output = model(input_image)

            # Obter a classe prevista
            _, predicted = torch.max(output, 1)
            predicted_class = predicted.item()

            conf = torch.softmax(output, dim=1)[0][predicted.item()].item()
            class_names = ['A', 'B', 'C']
            label = f"{class_names[predicted_class]}: {conf:.2f}"
            print(label)

        # Exiba o frame anotado
        cv2.imshow("Inference", annotated_frame)

        # Quebre o loop se 'q' for pressionado
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Libere os objetos de captura de vídeo e feche todas as janelas
for cap in captures:
    cap.release()
cv2.destroyAllWindows()