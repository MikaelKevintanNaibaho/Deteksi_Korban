import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

class CoordinateSubscriber(Node):
    def __init__(self):
        super().__init__('coordinate_subscriber')
        self.subscription = self.create_subscription(
            Point,
            'coordinate_korban',
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(
            'Received coordinate_korban: X={}, Y={}, Z={}'.format(
                msg.x, msg.y, msg.z
            )
        )

def main(args=None):
    rclpy.init(args=args)
    coordinate_subscriber = CoordinateSubscriber()
    rclpy.spin(coordinate_subscriber)
    coordinate_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
