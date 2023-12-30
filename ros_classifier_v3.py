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
    if not all(frame is not None for frame in frames_dict.values()):
        return

    resized_frames = [cv2.resize(frames_dict[key], (WIDTH, HEIGHT)) for key in sorted(frames_dict.keys())]
    combined_frame = np.hstack(tuple(resized_frames))
    cv2.imshow("Inference - Combined Cameras", combined_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        rospy.signal_shutdown("User terminated")

def process_image(msg, camera_name, publisher):
    global frames_dict
    frame = bridge.imgmsg_to_cv2(msg, "bgr8")
    res = yolo_model(frame)

    for i in range(len(res[0].conf)):
        if res[0].conf[i] > 0.75:
            bbox = res[0].boxes[i]
            x1, y1, x2, y2 = map(int, bbox[:4])

            # Cria a mensagem BoundingBox
            bbox_msg = BoundingBox()
            bbox_msg.Class = yolo_classes[int(res[0].cls[i])]
            bbox_msg.probability = res[0].conf[i]
            bbox_msg.xmin = x1
            bbox_msg.ymin = y1
            bbox_msg.xmax = x2
            bbox_msg.ymax = y2

            # Publica a mensagem
            publisher.publish(bbox_msg)

            # Extrai a ROI do frame
            roi = frame[y1:y2, x1:x2]
            result = letter_model(roi)

            # Anota o frame original com YOLO box, classe e confiança
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            yolo_label = f"{yolo_classes[int(res[0].cls[i])]}: {res[0].conf[i]:.2f}"
            letter_label = f"Letter {letter_classes[result[0].probs.top1]}: {result[0].probs.top1conf:.2f}"
            cv2.putText(frame, yolo_label, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, letter_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    frames_dict[camera_name] = frame

if __name__ == '__main__':
    rospy.init_node('image_listener_node')
    rospy.Subscriber("/usb_cam1/image_raw", Image, image_callback_1, queue_size=1)
    rospy.Subscriber("/usb_cam2/image_raw", Image, image_callback_2, queue_size=1)
    rospy.Subscriber("/usb_cam3/image_raw", Image, image_callback_3, queue_size=1)

    while not rospy.is_shutdown():
        combine_and_show_frames()
