import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import glob

# message_filters를 임포트합니다.
import message_filters

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.bridge = CvBridge()
        self.counter = 0

        # 저장 경로 설정
        self.rgb_dir = 'shared_output/rgb'
        self.depth_dir = 'shared_output/depth'
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)

        # 시작할 때 이전 파일 모두 삭제
        self.clear_images()
        self.get_logger().info("Previous images cleared.")

        # QoS 설정 (기존과 동일하게 사용하거나 상황에 맞게 ExactTime으로 변경 가능)
        qos = rclpy.qos.QoSProfile(
            depth=10,
            reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT
        )

        # 1. 각 토픽에 대한 구독자(Subscriber) 생성
        color_sub = message_filters.Subscriber(self, Image, '/image_rect', qos_profile=qos)
        depth_sub = message_filters.Subscriber(self, Image, '/depth', qos_profile=qos)

        # 2. TimeSynchronizer로 두 토픽을 묶어줍니다.
        # 타임스탬프가 거의 비슷한(slop=0.1초 이내) 메시지 쌍이 도착하면 self.sync_callback을 호출합니다.
        ts = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub],
            queue_size=10,
            slop=0.1  # 0.1초 이내의 타임스탬프 차이를 허용
        )
        ts.registerCallback(self.sync_callback)
        
        self.get_logger().info("Image saver node started, waiting for synchronized messages...")


    def clear_images(self):
        # .npy 파일도 삭제하도록 수정
        for f in glob.glob(os.path.join(self.rgb_dir, '*.png')) + glob.glob(os.path.join(self.rgb_dir, '*.npy')):
            os.remove(f)
        for f in glob.glob(os.path.join(self.depth_dir, '*.png')) + glob.glob(os.path.join(self.depth_dir, '*.npy')):
            os.remove(f)

    def sync_callback(self, color_msg, depth_msg):
        """
        하나의 동기화된 콜백에서 모든 작업을 처리합니다.
        """
        try:
            # 1. ROS 메시지를 OpenCV/Numpy 형식으로 변환 (최소한의 처리)
            color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough') # float32미터

            # 2. PNG 압축 대신 Numpy 배열을 그대로 저장 (.npy)
            # 이 작업은 CPU를 거의 사용하지 않아 매우 빠릅니다.
            rgb_filename = os.path.join(self.rgb_dir, f'{self.counter:06d}.npy')
            depth_filename = os.path.join(self.depth_dir, f'{self.counter:06d}.npy')
            
            np.save(rgb_filename, color_image)
            np.save(depth_filename, depth_image)
            
            self.get_logger().info(f'Successfully saved synchronized pair: {self.counter:06d}.npy')
            self.counter += 1

        except Exception as e:
            self.get_logger().error(f"Error processing message: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()