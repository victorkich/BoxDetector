#!/usr/bin/env python3

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np

def publish_random_image(publisher):
    # Create a random image
    random_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Convert the image to a ROS message using cv_bridge
    bridge = CvBridge()
    image_message = bridge.cv2_to_imgmsg(random_image, encoding="bgr8")

    # Publish the image
    publisher.publish(image_message)

def main():
    # Initialize the ROS node
    rospy.init_node('random_image_publisher')

    # Create publishers for each camera topic
    pub1 = rospy.Publisher('/usb_cam1/image_raw', Image, queue_size=10)
    pub2 = rospy.Publisher('/usb_cam2/image_raw', Image, queue_size=10)
    pub3 = rospy.Publisher('/usb_cam3/image_raw', Image, queue_size=10)

    rate = rospy.Rate(10) # 10Hz

    while not rospy.is_shutdown():
        publish_random_image(pub1)
        publish_random_image(pub2)
        publish_random_image(pub3)
        rate.sleep()

if __name__ == '__main__':
    main()
