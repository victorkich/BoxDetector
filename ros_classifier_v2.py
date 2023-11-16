#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
from ultralytics import YOLO

# Inicializa o modelo YOLO
yolo_model = YOLO('best.pt')

bridge = CvBridge()

# Dicionário para armazenar a última bounding box detectada e o contador de frames
last_boxes = {"Camera 1": {"box": None, "counter": 0, "smooth_box": None},
              "Camera 2": {"box": None, "counter": 0, "smooth_box": None},
              "Camera 3": {"box": None, "counter": 0, "smooth_box": None}}

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
    global last_boxes
    frame = bridge.imgmsg_to_cv2(msg, "bgr8")
    res = yolo_model(frame)

    boxes = res[0].boxes
    confidences = boxes.conf
    xyxys = boxes.xyxy

    if len(confidences) > 0 and max(confidences) > 0.75:
        higher_conf = np.argmax(confidences.cpu())
        box = xyxys[higher_conf].cpu().numpy().astype(int)

        if last_boxes[camera_name]["box"] is not None:
            # Suavização: Média entre a nova box e a box suavizada anterior
            alpha = 0.7  # Fator de suavização
            new_smooth_box = alpha * box + (1 - alpha) * last_boxes[camera_name]["smooth_box"]
            last_boxes[camera_name]["smooth_box"] = new_smooth_box
        else:
            last_boxes[camera_name]["smooth_box"] = box

        last_boxes[camera_name]["box"] = box
        last_boxes[camera_name]["counter"] = 20

    if last_boxes[camera_name]["counter"] > 0:
        smooth_box = last_boxes[camera_name]["smooth_box"].astype(int)
        x1, y1, x2, y2 = smooth_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        yolo_label = f"{yolo_classes[int(cls_idx)]}: {conf:.2f}" 
        letter_label = f"Letter {letter_classes[result[0].probs.top1]}: {result[0].probs.top1conf:.2f}"
        cv2.putText(frame, yolo_label, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, letter_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        last_boxes[camera_name]["counter"] -= 1

    frames_dict[camera_name] = frame

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
