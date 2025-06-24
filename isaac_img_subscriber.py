import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import glob
import message_filters

# message_filters를 임포트합니다.
import message_filters

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.bridge = CvBridge()
        self.counter = 0
        
        self.K = None
        self.caminfo_file = 'shared_output/cam_K.txt'

        self.rgb_dir = 'shared_output/rgb'
        self.depth_dir = 'shared_output/depth'
        
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)

        self.clear_images()
        self.get_logger().info("Previous images cleared.")

        qos = rclpy.qos.QoSProfile(
            depth=10,
            reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT
        )
        
        self.create_subscription(CameraInfo, '/camera_info_rect', self.caminfo_callback, qos)

        color_sub = message_filters.Subscriber(self, Image, '/image_rect', qos_profile=qos)
        depth_sub = message_filters.Subscriber(self, Image, '/depth', qos_profile=qos)

        ts = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub],
            queue_size=10,
            slop=0.1  
        )
        ts.registerCallback(self.sync_callback)
        
        self.get_logger().info("📷 Image saver node started. Waiting for messages...")


    def caminfo_callback(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
            np.savetxt(self.caminfo_file, self.K)
            self.get_logger().info(f"📸 CameraInfo received and saved to {self.caminfo_file}")

    def sync_callback(self, color_msg, depth_msg):
        try:
            color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough') 

            rgb_filename = os.path.join(self.rgb_dir, f'{self.counter:06d}.npy')
            depth_filename = os.path.join(self.depth_dir, f'{self.counter:06d}.npy')
            
            np.save(rgb_filename, color_image)
            np.save(depth_filename, depth_image)
            
            self.get_logger().info(f'Successfully saved synchronized pair: {self.counter:06d}.npy')
            self.counter += 1

        except Exception as e:
            self.get_logger().error(f"Error processing message: {e}")
            
    def clear_images(self):
        for f in glob.glob(os.path.join(self.rgb_dir, '*.png')) + glob.glob(os.path.join(self.rgb_dir, '*.npy')):
            os.remove(f)
        for f in glob.glob(os.path.join(self.depth_dir, '*.png')) + glob.glob(os.path.join(self.depth_dir, '*.npy')):
            os.remove(f)


def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()