#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from ktm_pede_detector_msgs.msg import BoundingBox

# Constants
ANGULAR_SPEED = 0.1
LINEAR_SPEED = 0.1
DETECTION_THRESHOLD = 0.7  # Adjust as needed
CAMERA_WIDTH = 480  # Adjust to match the width of your camera's image

# Global variables
current_target = 'green_box'  # Toggle between 'green_box' and 'blue_box'
letter_detected = None
boxes = {'left': None, 'front': None, 'right': None}

# ROS node initialization
rospy.init_node('box_approach_node')
cmd_vel_publisher = rospy.Publisher('/cmd_vel_source', Twist, queue_size=10)

def bounding_box_callback1(data):
    global boxes
    boxes['left'] = data.bounding_boxes

def bounding_box_callback2(data):
    global boxes
    boxes['front'] = data.bounding_boxes

def bounding_box_callback3(data):
    global boxes
    boxes['right'] = data.bounding_boxes

def process_boxes():
    """
    Processes the boxes from the three cameras and moves the robot.
    """
    global current_target, letter_detected, boxes

    target_box = get_target_box()
    if target_box:
        if target_box['camera'] == 'front':
            center_and_move_forward(target_box['box'])
        else:
            rotate_towards_box(target_box['camera'])
    else:
        rotate_robot()

def get_target_box():
    """
    Returns the box with the highest probability of the current target from the three cameras.
    """
    global current_target, boxes
    best_box = None
    for camera, bounding_boxes in boxes.items():
        # Check if bounding_boxes is not None before iterating
        if bounding_boxes:
            for box in bounding_boxes:
                if box.Class == current_target and box.probability >= DETECTION_THRESHOLD:
                    if not best_box or box.probability > best_box['box'].probability:
                        best_box = {'camera': camera, 'box': box}
    return best_box

def center_and_move_forward(box):
    """
    Centers the box on the front camera and moves the robot forward.
    """
    global letter_detected
    center_x = (box.xmin + box.xmax) / 2
    error = center_x - CAMERA_WIDTH / 2
    twist = Twist()

    if abs(error) > 20:  # Centering tolerance
        twist.angular.z = -ANGULAR_SPEED if error > 0 else ANGULAR_SPEED
    else:
        twist.linear.x = LINEAR_SPEED
        # Add logic to update `letter_detected`

    cmd_vel_publisher.publish(twist)

def rotate_towards_box(camera):
    """
    Rotates the robot in the direction of the camera where the box was detected.
    """
    twist = Twist()
    if camera == 'left':
        twist.angular.z = ANGULAR_SPEED
    elif camera == 'right':
        twist.angular.z = -ANGULAR_SPEED

    cmd_vel_publisher.publish(twist)

def rotate_robot():
    """
    Rotates the robot on its own axis.
    """
    twist = Twist()
    twist.angular.z = ANGULAR_SPEED
    cmd_vel_publisher.publish(twist)

def main():
    rospy.Subscriber('/camera1/bounding_boxes', BoundingBox, bounding_box_callback1)
    rospy.Subscriber('/camera2/bounding_boxes', BoundingBox, bounding_box_callback2)
    rospy.Subscriber('/camera3/bounding_boxes', BoundingBox, bounding_box_callback3)

    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        process_boxes()
        rate.sleep()

if __name__ == '__main__':
    main()
