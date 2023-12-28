#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from ktm_pede_detector_msgs.msg import BoundingBox

# Constantes
DISTANCE_MIN = 2.0  # Distância mínima da caixa (em metros)
SPEED_LINEAR = 0.5  # Velocidade linear para avançar em direção à caixa
SPEED_ANGULAR = 0.3 # Velocidade angular para centralizar com a caixa
BOX_TARGET = "Green Box"  # Começa procurando pela caixa verde

# Variáveis globais
current_box = None
current_letter = None

# Inicializa o nó
rospy.init_node('box_approach_node')
cmd_vel_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

def bounding_box_callback(msg):
    global current_box, current_letter
    if msg.Class == BOX_TARGET:
        current_box = msg
        current_letter = None  # Reset da letra ao detectar nova caixa
    elif msg.Class == "Letter":
        current_letter = msg

def approach_box():
    global current_box, current_letter, BOX_TARGET

    if current_box is None:
        return

    twist_msg = Twist()

    # Calcula a distância até o centro da imagem (assumindo que a largura da imagem é 640 pixels)
    center_x = (current_box.xmin + current_box.xmax) / 2
    error_x = center_x - 320

    # Ajusta a velocidade angular para centralizar a caixa
    twist_msg.angular.z = -error_x * SPEED_ANGULAR / 320

    # Verifica se a distância até a caixa é maior que a distância mínima
    if current_box.ymax < (480 - DISTANCE_MIN * 480):  # Assumindo que a altura da imagem é 480 pixels
        twist_msg.linear.x = SPEED_LINEAR

    cmd_vel_publisher.publish(twist_msg)

    # Checa se o robô está próximo o suficiente da caixa e se a letra foi detectada
    if current_box.ymax >= (480 - DISTANCE_MIN * 480) and current_letter:
        print(f"Letra na caixa {BOX_TARGET}: {current_letter.Class}")
        # Muda o alvo para a próxima caixa
        BOX_TARGET = "Blue Box" if BOX_TARGET == "Green Box" else "Green Box"
        current_box = None

if __name__ == '__main__':
    rospy.Subscriber('/camera1/bounding_boxes', BoundingBox, bounding_box_callback)
    rospy.Subscriber('/camera2/bounding_boxes', BoundingBox, bounding_box_callback)
    rospy.Subscriber('/camera3/bounding_boxes', BoundingBox, bounding_box_callback)

    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        approach_box()
        rate.sleep()
