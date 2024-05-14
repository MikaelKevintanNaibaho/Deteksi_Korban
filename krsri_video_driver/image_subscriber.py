import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
import cv2
import numpy as np
from .object_detection import ObjectDetection
from geometry_msgs.msg import Point


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__("image_subscriber")
        qos_profile = qos_profile_sensor_data
        self.subscription = self.create_subscription(
            CompressedImage,
            "camera/image_raw/compressed",
            self.listener_callback,
            qos_profile,
        )
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.modeldir = (
            "ros2_ws/src/krsri_video_driver/krsri_video_driver/custom_model_lite/"
        )

        self.object_detector = ObjectDetection(
            model_dir=self.modeldir, threshold=0.5, resolution="640x480"
        )

        self.victim_pub = self.create_publisher(Point, "coordinate_korban", 10);

    def listener_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        frame_detection, x, y, z = self.object_detector.perform_detection(frame)

        #publish koordinat korban di dalam frame untuk navigasi
        victim_point = Point()
        victim_point.x = round(float(x), 2)
        victim_point.y = round(float(y), 2)
        victim_point.z = round(float(z), 2)
        self.victim_pub.publish(victim_point)
        # Display the frame with OpenCV
        cv2.imshow("Object Detection", frame_detection)
        key = cv2.waitKey(1)
        if key == 27:  # Check if ESC key is pressed
            cv2.destroyAllWindows()  # Close all OpenCV windows
            self.get_logger().info("ESC key pressed. Shutting down...")
            self.destroy_node()
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    camera_subscriber = ImageSubscriber()
    rclpy.spin(camera_subscriber)
    camera_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
