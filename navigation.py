#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from ktm_pede_detector_msgs.msg import BoundingBox

# Constantes
SIZE_THRESHOLD = 5000  
SPEED_LINEAR = 0.1  
SPEED_ANGULAR = 0.1 
BOX_TARGET = "Green Box"  

# Variáveis globais
best_box = None
current_letter = None

# Inicializa o nó
rospy.init_node('box_approach_node')
cmd_vel_publisher = rospy.Publisher('/cmd_vel/', Twist, queue_size=10)

def bounding_box_callback(msg):
    global best_box, current_letter
    box_size = (msg.xmax - msg.xmin) * (msg.ymax - msg.ymin)
    if msg.Class == BOX_TARGET and (best_box is None or box_size > best_box[1]):
        best_box = (msg, box_size)
    elif msg.Class == "Letter":
        current_letter = msg

def approach_box():
    global best_box, current_letter, BOX_TARGET

    twist_msg = Twist()

    # Se nenhuma caixa foi detectada, gira em busca da caixa
    if best_box is None:
        twist_msg.angular.z = SPEED_ANGULAR
    else:
        bbox_msg, box_size = best_box
        center_x = (bbox_msg.xmin + bbox_msg.xmax) / 2
        error_x = center_x - 320

        # Centraliza com a caixa
        twist_msg.angular.z = -error_x * SPEED_ANGULAR / 320

        # Se a caixa for grande o suficiente, avança
        if box_size >= SIZE_THRESHOLD:
            twist_msg.linear.x = SPEED_LINEAR

    cmd_vel_publisher.publish(twist_msg)

    # Checa se está próximo o suficiente da caixa e a letra foi detectada
    if box_size >= SIZE_THRESHOLD and current_letter:
        print(f"Letra na caixa {BOX_TARGET}: {current_letter.Class}")
        BOX_TARGET = "Blue Box" if BOX_TARGET == "Green Box" else "Green Box"
        best_box = None
        current_letter = None

if __name__ == '__main__':
    rospy.Subscriber('/camera1/bounding_boxes', BoundingBox, bounding_box_callback)
    rospy.Subscriber('/camera2/bounding_boxes', BoundingBox, bounding_box_callback)
    rospy.Subscriber('/camera3/bounding_boxes', BoundingBox, bounding_box_callback)

    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        approach_box()
        rate.sleep()
