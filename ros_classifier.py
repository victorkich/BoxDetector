#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
from PIL import Image as PIL_Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ultralytics import YOLO

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

bridge = CvBridge()

# Dicionário global para armazenar frames processados de cada câmera
frames_dict = {"Camera 1": None, "Camera 2": None, "Camera 3": None}

def image_callback_1(msg):
    process_image(msg, "Camera 1")

def image_callback_2(msg):
    process_image(msg, "Camera 2")

def image_callback_3(msg):
    process_image(msg, "Camera 3")

def combine_and_show_frames():
    # Combina todos os frames em um único frame para exibição
    combined_frame = np.hstack(tuple(frames_dict.values()))

    cv2.imshow("Inference - Combined Cameras", combined_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        rospy.signal_shutdown("User terminated")

def process_image(msg, camera_name):
    frame = bridge.imgmsg_to_cv2(msg, "bgr8")
    res = yolo_model(frame)
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

        thresholded_in_pil = PIL_Image.fromarray(cv2.cvtColor(thresholded, cv2.COLOR_BGR2RGB))
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

    frames_dict[camera_name] = annotated_frame

    # Se todas as câmeras forneceram seus frames, mostramos o frame combinado
    if all(frame is not None for frame in frames_dict.values()):
        combine_and_show_frames()

if __name__ == '__main__':
    rospy.init_node('image_listener_node')

    # Subscreva nos tópicos de imagem ROS
    rospy.Subscriber("/usb_cam1/image_raw", Image, image_callback_1)
    rospy.Subscriber("/usb_cam2/image_raw", Image, image_callback_2)
    rospy.Subscriber("/usb_cam3/image_raw", Image, image_callback_3)

    # Mantenha o nó em execução até que seja fechado
    rospy.spin()
