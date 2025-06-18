import cv2
import numpy as np

class IsaacLabLiveReader:
    def __init__(self, rgb_dir="shared_output/rgb", depth_dir="shared_output/depth"):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.K = np.array([[615.0, 0, 320],
                           [0, 615.0, 240],
                           [0, 0, 1]], dtype=np.float32)  # 예시 intrinsics

    def get_color(self):
        img = cv2.imread(self.rgb_dir, cv2.IMREAD_COLOR)
        return img if img is not None else None

    def get_depth(self):
        depth_img = cv2.imread(self.depth_dir, cv2.IMREAD_UNCHANGED)
        if depth_img is not None:
            return depth_img.astype(np.float32) / 1000.0  # mm → m
        return None
