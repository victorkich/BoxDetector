#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from ktm_pede_detector_msgs.msg import BoundingBox

# Constants
SIZE_THRESHOLD = 5000  # Threshold for box size indicating proximity
SPEED_LINEAR = 0.1  # Linear speed to approach the box
SPEED_ANGULAR = 0.05  # Angular speed to align with the box
BOX_TARGET = "Green Box"  # Initially targeting the green box
IMAGE_WIDTH = 640  # Width of the camera image

# Global variables
best_box = None
current_letter = None

# Initialize the node
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

    # Calculate the center of the bounding box
    center_x = (bbox_msg.xmin + bbox_msg.xmax) / 2
    error_x = center_x - (IMAGE_WIDTH / 2)

    # Adjust angular velocity to align with the box
    twist_msg.angular.z = -error_x * SPEED_ANGULAR / (IMAGE_WIDTH / 2)

    # Check if the box is large enough (indicating proximity)
    if box_size < SIZE_THRESHOLD:
        twist_msg.linear.x = SPEED_LINEAR

    cmd_vel_publisher.publish(twist_msg)

    # Check if the robot is close enough to the box and if the letter has been detected
    if box_size >= SIZE_THRESHOLD and current_letter:
        rospy.loginfo(f"Letter in {BOX_TARGET}: {current_letter.Class}")
        # Switch target to the next box
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
