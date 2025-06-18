import os
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point


class PosePublisher(Node):
    def __init__(self):
        super().__init__('pose_publisher')

        self.pose_dir = 'debug/ob_in_cam'
        self.last_index = -1
        self.pose_pub = self.create_publisher(Point, '/object_position', 10)

        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info('PosePublisher started. Watching for new pose files...')

    def timer_callback(self):
        try:
            pose_files = sorted(f for f in os.listdir(self.pose_dir) if f.endswith('.txt'))
        except FileNotFoundError:
            self.get_logger().warn(f'Pose directory not found: {self.pose_dir}')
            return
    
        if not pose_files:
            return
    
        latest_file = pose_files[-1]
        frame_id = int(os.path.splitext(latest_file)[0])
    
        if frame_id <= self.last_index:
            return
    
        pose_path = os.path.join(self.pose_dir, latest_file)
        position = self.load_position_from_txt(pose_path)
        if position is not None:
            msg = Point()
            msg.x, msg.y, msg.z = position
            self.pose_pub.publish(msg)
            self.get_logger().info(f'Published pose for frame {frame_id}: ({msg.x:.3f}, {msg.y:.3f}, {msg.z:.3f})')
            self.last_index = frame_id

    def load_position_from_txt(self, path):
        try:
            pose = np.loadtxt(path)
            if pose.shape == (4, 4):
                return pose[:3, 3]  
        except Exception as e:
            self.get_logger().error(f'Failed to load pose from {path}: {e}')
        return None


def main(args=None):
    rclpy.init(args=args)
    node = PosePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
