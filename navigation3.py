#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from ktm_pede_detector_msgs.msg import BoundingBox

# Constants
ANGULAR_SPEED = 0.05
ANGULAR_SPEED_DYNAMIC = 0.05
LINEAR_SPEED = 0.1
DETECTION_THRESHOLD = 0.7  # Adjust as needed
CAMERA_WIDTH = 480  # Adjust to match the width of your camera's image

# Global variables
current_target = "Green Box"  # Toggle between 'green_box' and 'blue_box'
letter_detected = None
boxes = {'left': None, 'front': None, 'right': None}

# ROS node initialization
rospy.init_node('box_approach_node')
cmd_vel_publisher = rospy.Publisher('/cmd_vel_source', Twist, queue_size=10)

def process_boxes():
    """
    Processes the boxes from the three cameras and moves the robot.
    """
    global current_target, letter_detected, boxes

    target_box = get_target_box()
    if target_box:
        if target_box['camera'] == 'front':
            print("Centering and moving forward... Current target:", current_target)
            center_and_move_forward(target_box)
        else:
            print("Rotating towards box... Current target:", current_target)
            rotate_towards_box(target_box['camera'])
    else:
        print("Rotating the robot in order to find the target... Current target:", current_target)
        rotate_robot()

def bounding_box_callback1(data):
    global boxes
    boxes['front'] = data

def bounding_box_callback2(data):
    global boxes
    boxes['right'] = data

def bounding_box_callback3(data):
    global boxes
    boxes['left'] = data

def get_target_box():
    """
    Returns the box with the highest probability of the current target from the three cameras.
    """
    global current_target, boxes
    best_box = None
    for camera, bounding_box in boxes.items():
        # Check if bounding_box is not None before processing
        if bounding_box:
            box_size = (bounding_box.xmax - bounding_box.xmin) * (bounding_box.ymax - bounding_box.ymin)
            if bounding_box.Class == current_target and bounding_box.probability >= DETECTION_THRESHOLD:
                if not best_box or bounding_box.probability > best_box['box'].probability:
                    best_box = {'camera': camera, 'box': bounding_box, 'size': box_size}
    return best_box

def center_and_move_forward(box):
    """
    Centers the box on the front camera and moves the robot forward.
    """
    global current_target
    b_size = box['size']
    box = box['box']
    center_x = (box.xmin + box.xmax) / 4
    error = center_x - CAMERA_WIDTH / 2
    print("Center x:", center_x, "Error", error)
    twist = Twist()

    if abs(error) > 20:  # Centering tolerance
        twist.angular.z = -ANGULAR_SPEED if error > 0 else ANGULAR_SPEED
        print("Object is not centralized yet!")
        print("Turning " + "right ->" if twist.angular.z > 0 else "<- left")
    else:
        if b_size < 20000:
            twist.linear.x = LINEAR_SPEED
            print("Going forward!")
        else:
            current_target = "Blue Box" if current_target == "Green Box" else "Green Box"
            print("Updating target to ", current_target)

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
    ANGULAR_SPEED_DYNAMIC = twist.angular.z

    cmd_vel_publisher.publish(twist)

def rotate_robot():
    """
    Rotates the robot on its own axis.
    """
    twist = Twist()
    twist.angular.z = ANGULAR_SPEED_DYNAMIC
    cmd_vel_publisher.publish(twist)

def main():
    rospy.Subscriber('/camera1/bounding_boxes', BoundingBox, bounding_box_callback1)
    rospy.Subscriber('/camera2/bounding_boxes', BoundingBox, bounding_box_callback2)
    rospy.Subscriber('/camera3/bounding_boxes', BoundingBox, bounding_box_callback3)

    rate = rospy.Rate(4)  # 10 Hz
    while not rospy.is_shutdown():
        process_boxes()
        rate.sleep()

if __name__ == '__main__':
    main()
