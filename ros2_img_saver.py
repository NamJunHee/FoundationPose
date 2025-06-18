import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import time
import glob

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.bridge = CvBridge()

        qos = rclpy.qos.QoSProfile(
            depth=10,
            reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT
        )

        self.rgb_dir = 'shared_output/rgb'
        self.depth_dir = 'shared_output/depth'
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)

        # ✅ 시작할 때 .png 파일 모두 삭제
        self.clear_images()

        self.create_subscription(Image, '/image_rect', self.color_callback, qos)
        self.create_subscription(Image, '/depth', self.depth_callback, qos)

        self.counter = 0

    def clear_images(self):
        for f in glob.glob(os.path.join(self.rgb_dir, '*.png')):
            os.remove(f)
        for f in glob.glob(os.path.join(self.depth_dir, '*.png')):
            os.remove(f)

    def color_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        filename = os.path.join(self.rgb_dir, f'{self.counter:06d}.png')
        cv2.imwrite(filename, img)

    def depth_callback(self, msg):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')  # float32 (meter)
        depth_mm = (depth * 1000.0).astype(np.uint16)  # depth → millimeter, uint16로 압축
        filename = os.path.join(self.depth_dir, f'{self.counter:06d}.png')
        cv2.imwrite(filename, depth_mm)
        self.counter += 1


def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
