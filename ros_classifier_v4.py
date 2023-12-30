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
WIDTH = 480
HEIGHT = 360

bridge = CvBridge()

# Dicionário global para armazenar frames processados de cada câmera
# frames_dict = {"Camera 1": None, "Camera 2": None, "Camera 3": None}

frame1 = None
frame2 = None
frame3 = None

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
    # Redimensiona cada frame antes de combiná-los
    resized_frames = [cv2.resize(frame, (WIDTH, HEIGHT)) for frame in [frame1, frame2, frame3] if frame is not None]
    # resized_frames = [cv2.resize(frame, (WIDTH, HEIGHT)) for frame in frames_dict.values() if frame is not None]
    resized_frames = [resized_frames[2], resized_frames[0], resized_frames[1]]

    # Combina todos os frames redimensionados em um único frame para exibição
    combined_frame = np.hstack(tuple(resized_frames))

    cv2.imshow("Inference - Combined Cameras", combined_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        rospy.signal_shutdown("User terminated")


# Dicionário para armazenar a última bounding box e contador para cada câmera
last_bbox = {"Camera 1": {"bbox": None, "counter": 0},
             "Camera 2": {"bbox": None, "counter": 0},
             "Camera 3": {"bbox": None, "counter": 0}}



def process_image(msg, camera_name, publisher):
    # global frames_dict, last_bbox
    frame = bridge.imgmsg_to_cv2(msg, "bgr8")
    res = yolo_model(frame)

    boxes = res[0].boxes
    confidences = boxes.conf
    classes = boxes.cls
    xyxys = boxes.xyxy

    # Verifica se há alguma detecção com confiança suficiente
    detected = False
    if len(confidences) > 0 and max(confidences) > 0.75:
        higher_conf = np.argmax(confidences.cpu().numpy())
        box = xyxys[higher_conf]
        conf = confidences[higher_conf]
        cls_idx = classes[higher_conf]
        detected = True

        # Atualiza a bounding box e o contador
        last_bbox[camera_name] = {"bbox": box, "counter": 20, "cls_idx": cls_idx, "conf": conf}

    # Diminui o contador se não houver detecção
    if not detected and last_bbox[camera_name]["counter"] > 0:
        last_bbox[camera_name]["counter"] -= 1

    # Desenha a bounding box se o contador for maior que 0
    if last_bbox[camera_name]["counter"] > 0:
        box = last_bbox[camera_name]["bbox"]
        cls_idx = last_bbox[camera_name]["cls_idx"]
        conf = last_bbox[camera_name]["conf"]
        x1, y1, x2, y2 = map(int, box[:4])

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

    #frames_dict[camera_name] = frame
    if camera_name == "Camera 1":
        global frame1
        frame1 = frame
    elif camera_name == "Camera 2":
        global frame2
        frame2 = frame
    elif camera_name == "Camera 3":
        global frame3
        frame3 = frame

if __name__ == '__main__':
    rospy.init_node('image_listener_node')

    # Subscreva nos tópicos de imagem ROS
    rospy.Subscriber("/usb_cam1/image_raw", Image, image_callback_1, queue_size=1)
    rospy.Subscriber("/usb_cam2/image_raw", Image, image_callback_2, queue_size=1)
    rospy.Subscriber("/usb_cam3/image_raw", Image, image_callback_3, queue_size=1)

    # Mantenha o nó em execução até que seja fechado
    while not rospy.is_shutdown():
        # Se todas as câmeras forneceram seus frames, mostramos o frame combinado
        if all(frame is not None for frame in frames_dict.values()):
            combine_and_show_frames()
