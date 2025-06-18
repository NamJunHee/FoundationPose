import argparse
import glob
import logging
import os
import time

import cv2
import imageio
import numpy as np
import open3d as o3d
import trimesh
from estimater import (FoundationPose, ScorePredictor, PoseRefinePredictor,
                         set_logging_format, set_seed, depth2xyzmap, toOpen3dCloud,
                         draw_posed_3d_box, draw_xyz_axis)

try:
    import dr
except ImportError:
    print("Warning: dr (tiny-cuda-nn or nvisii) not found. GPU-accelerated renderer will not be available.")
    dr = None

class YcbineoatReader:

    def __init__(self, video_dir):
        self.video_dir = video_dir
        
        rgb_ids = set(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(self.video_dir, 'rgb', '*.npy')))
        depth_ids = set(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(self.video_dir, 'depth', '*.npy')))
        common_ids = sorted(list(rgb_ids.intersection(depth_ids)))

        self.id_strs = common_ids
        self.color_files = [os.path.join(self.video_dir, 'rgb', f'{fid}.npy') for fid in self.id_strs]
        self.depth_files = [os.path.join(self.video_dir, 'depth', f'{fid}.npy') for fid in self.id_strs]

        # self.K = np.array([[572.4114, 0., 325.2611],
        #                    [0., 573.5704, 242.0489],
        #                    [0., 0., 1.]])
        
        cam_k_file = os.path.join(video_dir, 'cam_K.txt')
        if not os.path.exists(cam_k_file):
            raise FileNotFoundError(f"cam_K.txt not found in {video_dir}")
        self.K = np.loadtxt(cam_k_file).reshape(3, 3)

    def get_color(self, i):
        bgr = np.load(self.color_files[i])
        color = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return color

    def get_depth(self, i):
        depth_m = np.load(self.depth_files[i])
        return depth_m
        
    def get_mask(self, i):
        mask_file = os.path.join(self.video_dir, 'masks', f'{self.id_strs[i]}.png')
        if not os.path.exists(mask_file):
            return None
        return cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

    def __len__(self):
        return len(self.id_strs)

def wait_for_data(data_dir, min_count=1, timeout=10):
    start = time.time()
    rgb_dir = os.path.join(data_dir, 'rgb')
    depth_dir = os.path.join(data_dir, 'depth')
    while True:
        rgb_files = glob.glob(os.path.join(rgb_dir, '*.npy'))
        depth_files = glob.glob(os.path.join(depth_dir, '*.npy'))
        if len(rgb_files) >= min_count and len(depth_files) >= min_count:
            print(f"✅ Found {len(rgb_files)} RGB and {len(depth_files)} data files.")
            break
        if time.time() - start > timeout:
            raise RuntimeError("⏰ Timeout: 데이터 파일이 충분히 생성되지 않았습니다.")
        print(f"⏳ Waiting for data files... RGB: {len(rgb_files)}, Depth: {len(depth_files)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
    parser.add_argument('--test_scene_dir', type=str, default='/home/nmail-njh/NMAIL/01_Project/Robot_Grasping/FoundationPose/shared_output')
    parser.add_argument('--est_refine_iter', type=int, default=10)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    parser.add_argument('--fps', type=int, default=15, help='Target FPS for visualization')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    print("Clearing previous .npy files...")
    rgb_dir_to_clear = os.path.join(args.test_scene_dir, 'rgb')
    depth_dir_to_clear = os.path.join(args.test_scene_dir, 'depth')
    for f in glob.glob(os.path.join(rgb_dir_to_clear, '*.npy')):
        os.remove(f)
    for f in glob.glob(os.path.join(depth_dir_to_clear, '*.npy')):
        os.remove(f)
    print(f"✅ Cleared previous .npy files from {args.test_scene_dir}")

    print("Clearing previous mask files...")
    masks_dir_to_clear = os.path.join(args.test_scene_dir, 'masks')
    if os.path.exists(masks_dir_to_clear):
        for f in glob.glob(os.path.join(masks_dir_to_clear, '*.png')):
            os.remove(f)
        print(f"✅ Cleared previous mask files from {masks_dir_to_clear}")
    else:
        print("ℹ️  Masks directory not found, skipping cleanup.")

    mesh = trimesh.load(args.mesh_file)
    mesh.vertices = mesh.vertices.astype(np.float32)
    mesh.vertex_normals = mesh.vertex_normals.astype(np.float32)

    debug = args.debug
    debug_dir = args.debug_dir
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext() if dr else None
    if not glctx:
        print("Warning: CUDA rasterizer context not available. Visualization might be affected.")

    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=debug,
        glctx=glctx
    )
    logging.info("Estimator initialization done")

    wait_for_data(args.test_scene_dir, min_count=1, timeout=20)
    
    TARGET_FPS = args.fps
    TARGET_FRAME_TIME = 1.0 / TARGET_FPS
    
    frame_idx = 0
    pose = None

    while True:
        loop_start_time = time.time()
        try:
            reader = YcbineoatReader(video_dir=args.test_scene_dir)
            
            if frame_idx < len(reader):
                logging.info(f'Processing frame: {frame_idx} ({reader.id_strs[frame_idx]})')
                color = reader.get_color(frame_idx)
                depth = reader.get_depth(frame_idx)
                
                if frame_idx == 0:
                    mask = reader.get_mask(frame_idx)
                    
                    if mask is None:
                        logging.info("No mask found. Generating mask by removing the ground plane with RANSAC.")
                        
                        height, width = depth.shape
                        o3d_depth = o3d.geometry.Image(depth)
                        
                        pcd = o3d.geometry.PointCloud.create_from_depth_image(
                            o3d_depth,
                            o3d.camera.PinholeCameraIntrinsic(width, height, reader.K[0,0], reader.K[1,1], reader.K[0,2], reader.K[1,2]),
                            depth_scale=1.0, depth_trunc=3.0
                        )
                        
                        if not pcd.has_points():
                            logging.warning("Could not create point cloud. Falling back to use the whole image.")
                            mask = np.ones_like(depth, dtype=bool)
                        else:
                            plane_model, inliers = pcd.segment_plane(distance_threshold=0.015,
                                                                     ransac_n=3,
                                                                     num_iterations=1000)
                            
                            num_outliers = len(pcd.points) - len(inliers)
                            
                            if num_outliers < 100:
                                 logging.warning(f"남은 점의 개수({num_outliers})가 100개 미만이므로 마스크 생성 실패로 간주합니다.")
                                 mask = np.ones_like(depth, dtype=bool)
                            else:
                                logging.info(f"남은 점의 개수({num_outliers})가 충분하여 마스크를 생성하고 저장합니다.")
                                
                                mask = np.zeros((height, width), dtype=bool)
                                
                                all_indices = np.arange(len(pcd.points))
                                outlier_indices = np.where(np.isin(all_indices, inliers, invert=True))[0]
                                
                                outlier_pts = np.asarray(pcd.points)[outlier_indices]
                                z_vals = outlier_pts[:, 2]
                                valid_z_mask = (z_vals > 0.1) & (z_vals < 1.5)
                                filtered_indices = outlier_indices[valid_z_mask]
                                
                                rows = filtered_indices // width
                                cols = filtered_indices % width
                                
                                mask[rows, cols] = True

                                # 파일 저장을 위해 0/255 값을 갖는 이미지로 변환
                                mask_to_save = (mask.astype(np.uint8)) * 255
                                mask_dir = os.path.join(args.test_scene_dir, 'masks')
                                os.makedirs(mask_dir, exist_ok=True)
                                save_path = os.path.join(mask_dir, f'{reader.id_strs[frame_idx]}.png')
                                
                                try:
                                    success = cv2.imwrite(save_path, mask_to_save)
                                    if success:
                                        logging.info(f"✅ Automatically generated mask saved to {save_path}")
                                    else:
                                        logging.error(f"❌ Failed to save mask to {save_path}. Check file path and permissions.")
                                except Exception as e:
                                    logging.error(f"❌ An error occurred while saving the mask: {e}")
                                
                                cv2.imshow('Generated Mask', mask_to_save)
                                cv2.waitKey(1)
                    else:
                        logging.info(f"✅ Found existing mask: {reader.id_strs[frame_idx]}.png. Using it for registration.")
                        mask = mask.astype(bool)

                    pose = est.register(
                        K=reader.K, rgb=color, depth=depth,
                        ob_mask=mask, iteration=args.est_refine_iter
                    )
                else:
                    if cv2.getWindowProperty('Generated Mask', cv2.WND_PROP_VISIBLE) >= 1:
                        cv2.destroyWindow('Generated Mask')

                    pose = est.track_one(
                        rgb=color, depth=depth, K=reader.K,
                        iteration=args.track_refine_iter
                    )

                if pose is None:
                    logging.error(f"Tracking failed for frame {frame_idx}. Re-initializing...")
                    frame_idx = 0
                    continue

                os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
                np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[frame_idx]}.txt', pose.reshape(4, 4))

                if debug >= 1:
                    center_pose = pose @ np.linalg.inv(to_origin)
                    vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
                    vis = draw_posed_3d_box(reader.K, img=vis, ob_in_cam=center_pose, bbox=bbox)
                    cv2.imshow('pose tracking', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Terminating...")
                        break

                if debug >= 2:
                    os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
                    imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[frame_idx]}.png', vis)
                
                frame_idx += 1
            else:
                print(f"⏳ Waiting for new frames... (Processed: {frame_idx})", end='\r')

            elapsed_time = time.time() - loop_start_time
            wait_time = TARGET_FRAME_TIME - elapsed_time
            if wait_time > 0:
                time.sleep(wait_time)

        except KeyboardInterrupt:
            print("\nTerminating...")
            break
        except Exception as e:
            logging.error(f"An error occurred in the main loop: {e}")
            time.sleep(1)

    cv2.destroyAllWindows()