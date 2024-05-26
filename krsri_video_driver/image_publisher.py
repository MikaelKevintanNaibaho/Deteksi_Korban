import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2


class ImagePublisher(Node):
    def __init__(self):
        super().__init__("image_publisher")
        self.publisher_ = self.create_publisher(
            CompressedImage, "camera/image_raw/compressed", 10
        )
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(3)
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera.")
            return

        self.timer = self.create_timer(
            0.033, self.publish_camera
        )  # Adjust the publishing rate to 60 Hz

    def publish_camera(self):
        ret, frame = self.cap.read()
        if ret:
            # Resize the frame to 320x240
            frame_resized = cv2.resize(frame, (640, 480))
            # Compress the frame
            compressed_frame = cv2.imencode(".jpg", frame_resized)[1].tostring()
            # Create a CompressedImage message
            image_msg = CompressedImage()
            image_msg.header.stamp = self.get_clock().now().to_msg()
            image_msg.format = "jpeg"
            image_msg.data = compressed_frame
            self.publisher_.publish(image_msg)
            self.get_logger().info("Image published")

            coordinate_msg = String()
            


def main(args=None):
    rclpy.init(args=args)
    camera_publisher = ImagePublisher()
    rclpy.spin(camera_publisher)
    camera_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
