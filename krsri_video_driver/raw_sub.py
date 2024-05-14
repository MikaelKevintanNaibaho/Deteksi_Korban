#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import threading


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__("raw_sub")
        # subscribe to compressed image so it save the bandwidth
        self.subscription = self.create_subscription(
            CompressedImage, "camera/image_raw/compressed", self.image_callback, 10
        )
        self.subscription  # prevent unused variable warning

        self.bridge = CvBridge()
        self.window_name = "Image Subscriber"
        cv2.namedWindow(self.window_name)

        # Create a threading Event to control the display loop
        self.display_event = threading.Event()

    def image_callback(self, msg):
        # Convert the ROS Image message to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Start a new thread for image display
        display_thread = threading.Thread(target=self.display_image, args=(cv_image,))
        display_thread.start()

    def display_image(self, cv_image):
        # Display the image asynchronously
        cv2.imshow(self.window_name, cv_image)
        cv2.waitKey(1)  # Display the image for 1 millisecond

        # Set the threading Event to indicate that the image has been displayed
        self.display_event.set()


def main(args=None):
    rclpy.init(args=args)
    camera_subscriber = ImageSubscriber()
    rclpy.spin(camera_subscriber)
    camera_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
