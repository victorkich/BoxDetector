#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from ktm_pede_detector_msgs.msg import BoundingBox

# Constantes
SIZE_THRESHOLD = 5000  # Limiar para o tamanho do retângulo que indica proximidade (ajuste conforme necessário)
SPEED_LINEAR = 0.1  # Velocidade linear para avançar em direção à caixa
SPEED_ANGULAR = 0.1 # Velocidade angular para centralizar com a caixa
BOX_TARGET = "Green Box"  # Inicialmente procurando pela caixa verde

# Variáveis globais
best_box = None
current_letter = None

# Inicializa o nó
rospy.init_node('box_approach_node')
cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

def bounding_box_callback(msg):
    global best_box, current_letter
    box_size = (msg.xmax - msg.xmin) * (msg.ymax - msg.ymin)
    if msg.Class == BOX_TARGET and (best_box is None or box_size > best_box[1]):
        best_box = (msg, box_size)
    elif msg.Class == "Letter":
        current_letter = msg

def approach_box():
    global best_box, current_letter, BOX_TARGET

    if best_box is None:
        return

    twist_msg = Twist()
    bbox_msg, box_size = best_box

    # Calcula o centro da bounding box
    center_x = (bbox_msg.xmin + bbox_msg.xmax) / 2
    error_x = center_x - 320  # Assumindo que a largura da imagem é 640 pixels

    # Ajusta a velocidade angular para centralizar a caixa
    twist_msg.angular.z = -error_x * SPEED_ANGULAR / 320

    # Verifica se a caixa é grande o suficiente (indicando proximidade)
    if box_size < SIZE_THRESHOLD:
        twist_msg.linear.x = SPEED_LINEAR

    cmd_vel_publisher.publish(twist_msg)

    # Checa se o robô está próximo o suficiente da caixa e se a letra foi detectada
    if box_size >= SIZE_THRESHOLD and current_letter:
        print(f"Letra na caixa {BOX_TARGET}: {current_letter.Class}")
        # Muda o alvo para a próxima caixa
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
