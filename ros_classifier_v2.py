#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from ktm_pede_detector_msgs.msg import BoundingBox

from cv_bridge import CvBridge

import cv2
import numpy as np
from ultralytics import YOLO

# Inicializa o modelo YOLO
yolo_model = YOLO('best.pt')
letter_model = YOLO('letter.pt')

yolo_classes = {0: "Blue Box", 1: "Green Box"}
letter_classes = {0: "A", 1: "B", 2: "C"}

bridge = CvBridge()

# Dicionário global para armazenar frames processados de cada câmera
frames_dict = {"Camera 1": None, "Camera 2": None, "Camera 3": None}

# Publishers para cada câmera
pub_bbox1 = rospy.Publisher('/camera1/bounding_boxes', BoundingBox, queue_size=10)
pub_bbox2 = rospy.Publisher('/camera2/bounding_boxes', BoundingBox, queue_size=10)
pub_bbox3 = rospy.Publisher('/camera3/bounding_boxes', BoundingBox, queue_size=10)

def image_callback_1(msg):
    process_image(msg, "Camera 1", pub_bbox1)

def image_callback_2(msg):
    process_image(msg, "Camera 2", pub_bbox2)

def image_callback_3(msg):
    process_image(msg, "Camera 3", pub_bbox3)

def combine_and_show_frames():
    # Combina todos os frames em um único frame para exibição
    combined_frame = np.hstack(tuple(frames_dict.values()))

    cv2.imshow("Inference - Combined Cameras", combined_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        rospy.signal_shutdown("User terminated")

# Dicionário para armazenar a última bounding box e contador para cada câmera
last_bbox = {"Camera 1": {"bbox": None, "counter": 0},
             "Camera 2": {"bbox": None, "counter": 0},
             "Camera 3": {"bbox": None, "counter": 0}}

def process_image(msg, camera_name, publisher):
    global frames_dict, last_bbox
    frame = bridge.imgmsg_to_cv2(msg, "bgr8")
    res = yolo_model(frame)

    boxes = res[0].boxes
    confidences = boxes.conf
    classes = boxes.cls
    xyxys = boxes.xyxy

    detected = False
    cls_idx = None
    conf = None

    if len(confidences) > 0 and max(confidences) > 0.75:
        higher_conf = np.argmax(confidences.cpu().numpy())
        box = xyxys[higher_conf]
        conf = confidences[higher_conf]
        cls_idx = classes[higher_conf]
        detected = True

    if detected:
        # Atualiza a bounding box e o contador
        last_bbox[camera_name]["bbox"] = box
        last_bbox[camera_name]["counter"] = 20  # Reset para 20 frames
    else:
        # Diminui o contador se não houver detecção
        if last_bbox[camera_name]["counter"] > 0:
            last_bbox[camera_name]["counter"] -= 1

    # Desenha a bounding box se o contador for maior que 0
    if last_bbox[camera_name]["counter"] > 0:
        box = last_bbox[camera_name]["bbox"]
        x1, y1, x2, y2 = map(int, box[:4])

        if cls_idx is not None and conf is not None:
            # Creating the BoundingBox Message
            bbox_msg = BoundingBox()
            bbox_msg.Class = yolo_classes[int(cls_idx)]
            bbox_msg.probability = conf
            bbox_msg.xmin = x1
            bbox_msg.ymin = y1
            bbox_msg.xmax = x2
            bbox_msg.ymax = y2

            # Publishing
            publisher.publish(bbox_msg)

            # Extract ROI from the frame
            roi = frame[y1:y2, x1:x2]
            
            result = letter_model(roi)

            # Annotate original frame with YOLO box, class, and confidence
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            yolo_label = f"{yolo_classes[int(cls_idx)]}: {conf:.2f}" 
            letter_label = f"Letter {letter_classes[result[0].probs.top1]}: {result[0].probs.top1conf:.2f}"
            cv2.putText(frame, yolo_label, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, letter_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    frames_dict[camera_name] = frame


if __name__ == '__main__':
    rospy.init_node('image_listener_node')

    # Subscreva nos tópicos de imagem ROS
    rospy.Subscriber("/usb_cam1/image_raw", Image, image_callback_1)
    rospy.Subscriber("/usb_cam2/image_raw", Image, image_callback_2)
    rospy.Subscriber("/usb_cam3/image_raw", Image, image_callback_3)

    # Mantenha o nó em execução até que seja fechado
    while not rospy.is_shutdown():
        # Se todas as câmeras forneceram seus frames, mostramos o frame combinado
        if all(frame is not None for frame in frames_dict.values()):
            combine_and_show_frames()
