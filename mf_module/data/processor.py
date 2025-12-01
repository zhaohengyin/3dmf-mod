import os 
import numpy as np
from pathlib import Path 
import math 
import socket 
import cv2
import random
from mf_module.data.image_noise import get_random_boundary, get_random_internal, bernoulli_pm1, area_to_boundary_ratio
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
import matplotlib.pyplot as plt 


def visualize_numpy_field(field, names, output_path, n_column=4, bound={}):
    channels = field.shape[0]
    if channels == 1:
        plt.imshow(field[0], cmap='viridis', aspect='auto')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(names[0] if 0 < len(names) else f"Channel")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    else:
        n_rows = (channels + n_column - 1) // n_column  # Ceiling division for rows

        fig, axes = plt.subplots(n_rows, n_column, figsize=(4 * n_column, 4 * n_rows))
        axes = axes.flatten() if channels > 1 else [axes]

        for idx in range(len(axes)):
            ax = axes[idx]
            if idx < channels:
                if names[idx] in bound:
                    im = ax.imshow(field[idx], cmap='viridis', aspect='equal', vmin=bound[names[idx]][0], vmax=bound[names[idx]][1])
                else:
                    im = ax.imshow(field[idx], cmap='viridis', aspect='equal')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(names[idx] if idx < len(names) else f"Channel {idx}")
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)


def get_random_direction():
    x = np.random.randn(3)
    x = x / np.linalg.norm(x)
    return x 


def get_rotation(axis, angle):
    rotvec = axis * angle
    rotation_matrix = R.from_rotvec(rotvec).as_matrix()
    return rotation_matrix


def random_small_rotation(degrees=10.0):
    """
    Generate a small random 3D rotation matrix (within ±degrees).

    Args:
        degrees (float): Maximum absolute rotation angle in degrees.

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    axis = get_random_direction()
    max_rad = np.deg2rad(degrees)
    angle = np.random.uniform(-max_rad, max_rad)

    # Convert to rotation matrix using scipy
    rotvec = axis * angle
    rotation_matrix = R.from_rotvec(rotvec).as_matrix()

    return rotation_matrix


def apply_transform_on_points(transform, points):
    '''
        transform: [4, 4], np.ndarray
        points: [n, 3], np.ndarray
    '''
    n = points.shape[0]
    points_homogeneous = np.hstack([points, np.ones((n, 1))])  # [n, 4]
    transformed_points_homogeneous = points_homogeneous @ transform.T  # [n, 4]
    transformed_points = transformed_points_homogeneous[:, :3] / transformed_points_homogeneous[:, 3][:, np.newaxis]
    return transformed_points


class MotionDataProcessor:
    """
        Process each training data. Reformat the structure and use randomizers to add some noises.
    """
    
    def __init__(self, randomizer=None, static_camera=True, camera_rotation_scale=45, rescale_size=None):
        self.randomizer = randomizer
        self.static_camera = static_camera
        self.camera_rotation_scale = camera_rotation_scale
        self.rescale_size = rescale_size
        return 

    def get_new_camera_pose(self):
        next_pose = np.eye(4)
        if np.random.uniform(0, 1, 1)[0] < 0.5:
            # with some probability, do not do any rotation.
            rot = np.exp(np.random.uniform(-5, 0, 1)[0] * np.log(10)) * 5
            next_pose[:3, :3] = random_small_rotation(np.random.uniform(-rot, rot, 1)[0])   
        else:    
            next_pose[:3, :3] = random_small_rotation(np.random.uniform(-self.camera_rotation_scale, self.camera_rotation_scale, 1)[0])   

        moving_direction = np.random.randn(3)
        moving_direction = moving_direction / (np.linalg.norm(moving_direction) + 1e-8)

        if np.random.uniform(0, 1, 1)[0] < 0.5:
            translation = moving_direction * np.exp(np.random.uniform(-5, 0, 1)[0] * np.log(10)) * 0.01
        else:
            translation = moving_direction * np.exp(np.random.uniform(-2, 0, 1)[0] * np.log(10)) * 0.03

        next_pose[:3, 3] = translation
        return next_pose

    def generate_data(self, data):
        """
            This function synthesize flow data given the input
        """
        # Note, we use a sparse vector to store depth.
        depth = data["depth"]   # [N, ]
        h, w = data["hw"]       
        fovy = data["fovy"]
        pixel_x = data["px"]    # [N, ]
        pixel_y = data["py"]    # [N, ]
        cx = w / 2
        cy = h / 2 

        f = h / 2 / math.tan(fovy)

        cam_x = (pixel_x + 0.5 - cx) / f * depth 
        cam_y = - (pixel_y + 0.5 - cy) / f * depth 
        cam_z = - depth 

        pixel_3d = np.stack([pixel_x.astype(np.float32) + 0.5, pixel_y.astype(np.float32) + 0.5, depth], axis=-1)
        points_3d = np.stack([cam_x, cam_y, cam_z], axis=-1)

        if len(points_3d) <= 5:
            return None 

        diameter = np.linalg.norm(np.max(points_3d, axis=0) - np.min(points_3d, axis=0)) / 2
        points_3d_mean = np.mean(points_3d, axis=0)

        current_pose = np.eye(4)
        current_pose[:3, 3] = points_3d_mean

        # Do a small rotation about the rotation center.
        rotation_center_dir = get_random_direction()
        if np.random.uniform(0, 1, 1)[0] < 0.2:
            rotation_center_dist = np.random.uniform(0, 1, 1)[0]
        else:
            rotation_center_dist =  np.random.uniform(0, 0.03, 1)[0]

        rotation_center = current_pose[:3, 3] + rotation_center_dir * rotation_center_dist

        rotation_axis = get_random_direction()
        if np.random.uniform(0, 1, 1)[0] < 0.8:
            r = max([rotation_center_dist, diameter])

            if np.random.uniform(0, 1, 1)[0] < 0.5:
                desired_rotation_distance = np.exp(np.random.uniform(-1, 0, 1)[0] * np.log(10)) * 0.05
            else:
                desired_rotation_distance = np.random.uniform(0.001, 0.005, 1)[0]
            if desired_rotation_distance > 3.14 / 4 * r:
                desired_rotation_distance = 3.14 / 4 * r # clamp at pi/4.

            rotation_angle = desired_rotation_distance / (r + 1e-4)

        else:
            rotation_angle = 0

        # Rotation about rotation center.
        rotation_matrix = get_rotation(rotation_axis, rotation_angle)
        T0 = np.eye(4)
        T0[:3, 3] = -rotation_center
        
        T1 = np.eye(4)
        T1[:3, :3] = rotation_matrix

        T2 = np.eye(4)
        T2[:3, 3] = rotation_center

        next_pose = T2 @ T1 @ T0 @ current_pose
        

        # Do a small translation
        if np.random.uniform(0, 1, 1)[0] < 0.5:
            translation = np.exp(np.random.uniform(-1, 0, 1)[0] * np.log(10)) * 0.03 * rotation_axis
        else:
            translation = np.exp(np.random.uniform(-3, 0, 1)[0] * np.log(10)) * 0.01 * rotation_axis
        
        next_pose[:3, 3] += translation

        transform = next_pose @ np.linalg.inv(current_pose)
        
        # Rx + t; x: col vec.
        # xR^T + t^T: x: row vec
        points_3d_new = points_3d @ transform[:3, :3].transpose() + transform[:3, 3]  # X_cam_points. This is how things are like in the source camera. 

        next_camera_pose = np.eye(4)

        if not self.static_camera:
            next_camera_pose = self.get_new_camera_pose() # T_cam_newcam
            points_3d_new_in_new_cam = apply_transform_on_points(np.linalg.inv(next_camera_pose), points_3d_new)
        
        else:
            points_3d_new_in_new_cam = points_3d_new

        new_pixel_x = points_3d_new_in_new_cam[:, 0] / (-points_3d_new_in_new_cam[:, 2]) * f + cx
        new_pixel_y = - points_3d_new_in_new_cam[:, 1] / (-points_3d_new_in_new_cam[:, 2]) * f + cy 
        new_pixel_z = - points_3d_new_in_new_cam[:, 2]
        new_pixel_3d = np.stack([new_pixel_x, new_pixel_y, new_pixel_z], axis=-1)

        motion_flow = points_3d_new - points_3d
        pixel_flow = new_pixel_3d - pixel_3d

        depth_image = np.zeros((h, w)).astype(np.float32)
        depth_image[pixel_y, pixel_x] = depth 

        mask_image = np.zeros((h, w)).astype(np.float32)
        mask_image[pixel_y, pixel_x] = 1.0

        motion_image = np.zeros((h, w, 3)).astype(np.float32)
        motion_image[pixel_y, pixel_x] = motion_flow 

        pixel_flow_image = np.zeros((h, w, 3)).astype(np.float32)
        pixel_flow_image[pixel_y, pixel_x] = pixel_flow 

        return {
            "mask": mask_image, 
            "pf": pixel_flow_image, 
            "mf": motion_image, 
            "depth": depth_image[..., None],
            "fovy": data["fovy"],
            "p": points_3d,
            "next_p": points_3d_new,
            "cam_t": next_camera_pose
        }

    def rescale_pixelflow(self, x):
        x = np.array(x) # create a copy
        x[:, :, :2] = x[:, :, :2] / 10.0
        x[:, :, 2] = x[:, :, 2] * 20.0  # z_channel has to be amplified.
        return x 

    def rescale_motionflow(self, x, wrap_gt=True):
        if not wrap_gt:
            return x / 20.0     # this is gt (in meters.)
        return x * 20.0         # rescaled gt.

    def parse_realworld_data(self, data):
        def bilinear_sample(image, x, y):
            """
            image: H x W array
            x, y: float coordinates (shape: N)
            returns sampled values at those coords using bilinear interpolation
            """
            H, W = image.shape

            x0 = np.floor(x).astype(np.int32)
            x1 = np.clip(x0 + 1, 0, W - 1)
            y0 = np.floor(y).astype(np.int32)
            y1 = np.clip(y0 + 1, 0, H - 1)

            wx = x - x0
            wy = y - y0

            Ia = image[y0, x0]
            Ib = image[y0, x1]
            Ic = image[y1, x0]
            Id = image[y1, x1]

            return (
                Ia * (1 - wx) * (1 - wy) +
                Ib * (wx)     * (1 - wy) +
                Ic * (1 - wx) * (wy)     +
                Id * (wx)     * (wy)
            )

        depth = np.squeeze(data['depth'])
        next_depth = np.squeeze(data['next_depth'])
        mask = np.squeeze(data['mask'])
        camera_intrinsics = data["camera_intrinsics"]
        camera_transform = data.get("camera_transform", np.eye(4))
        pixel_trajectory = data['pt']
        
        H, W = mask.shape[0], mask.shape[1]
        if pixel_trajectory.shape[1] == 0:
            return None 

        sample_idx = np.random.randint(0, pixel_trajectory.shape[1]-1, max([400, int(pixel_trajectory.shape[1] * 0.005)]))

        pixel_trajectory = pixel_trajectory[:, sample_idx, :]

        pixel_flow_init = pixel_trajectory[0]
        pixel_flow_begin_x = pixel_trajectory[0, :, 0].astype(np.int32)
        pixel_flow_begin_y = pixel_trajectory[0, :, 1].astype(np.int32)
       
        pixel_flow_end_x = pixel_trajectory[1, :, 0].astype(np.int32)
        pixel_flow_end_y = pixel_trajectory[1, :, 1].astype(np.int32)
        
        pixel_flow_end_x = np.clip(pixel_flow_end_x, 0, W-1)
        pixel_flow_end_y = np.clip(pixel_flow_end_y, 0, H-1)
        pixel_flow = pixel_trajectory[1] - pixel_trajectory[0]  # order: x, y
        depth_flow = bilinear_sample(next_depth, pixel_flow_end_x, pixel_flow_end_y) - bilinear_sample(depth, pixel_flow_begin_x, pixel_flow_begin_y)
        
        invalid_mask = np.abs(depth_flow) > 0.1
        depth_flow[invalid_mask] = -100.0

        # [N, 3]
        pixel_flow3d = np.concatenate([pixel_flow.reshape(-1, 2), depth_flow.reshape(-1, 1)], axis=-1) # order, x,y, depth

        # Render to images
        pixel_mask = np.zeros((H, W))
        pixel_flow_image = np.zeros((H, W, 3))

        pixel_mask[pixel_flow_begin_y, pixel_flow_begin_x] = 1.0
        pixel_flow_image[pixel_flow_begin_y, pixel_flow_begin_x] = pixel_flow3d
        invalid_mask = pixel_flow_image[:, :, 2] < -10.0
        pixel_flow_image[invalid_mask, 2] = 0.0
        pixel_flow_image[:, :, 2] = np.clip(pixel_flow_image[:, :, 2], -0.1, 0.1)
        pixel_mask[invalid_mask] = 0.0
        x = self.wrap_input(mask[..., None], (mask * depth)[..., None], pixel_mask[..., None], pixel_flow_image)

        out = {
            "x": x.astype(np.float32),
            "cam": camera_intrinsics.astype(np.float32), #[fx, fy, cx, cy]
            "cam_t": self.flatten_transform(camera_transform).astype(np.float32),
            "mask": mask[None].astype(np.float32), 
            "pixmask": pixel_mask[None].astype(np.float32)
        }
        return out

    def parse_representation_output(self, output, data=None):
        B, C, H, W = output["motion"].shape 
        depth = output["motion"][:, :1, ...]
        motion_flow = self.rescale_motionflow(output["motion"][:, 1:, ...], wrap_gt=False)

        out = np.concatenate([depth, motion_flow], axis=1)
        if data is not None:
            mask = data.get("mask", None)
            if mask is not None:
                mask = np.squeeze(mask)[None, None, ...]
                out = out * mask
        output["motion"] = out
        return output

    def wrap_input(self, mask, depth, pixel_mask, pixel_flow):
        pixel_flow = self.rescale_pixelflow(pixel_flow)
        x = np.concatenate((mask, depth, pixel_mask, pixel_flow), axis=-1)
        x = np.moveaxis(x, -1, 0)  # channel first.
        return x.astype(np.float32)

    def wrap_output(self, depth, motionflow):
        motionflow = self.rescale_motionflow(motionflow)
        y = np.concatenate((depth, motionflow), axis=-1)
        y = np.moveaxis(y, -1, 0)  # channel first.
        return y.astype(np.float32)

    def flatten_transform(self, t):
        return np.concatenate((t[:3, :3].reshape(-1), t[:3, 3]))
        
    def process_dataset(self, data):
        """ 
            This function do some necessary data augmentations.
        """
        if "mf" not in data:
            data = self.generate_data(data)
            if data is None:
                return None 

        mask = data['mask']
        pixelflow = data['pf']
        depth = data['depth']
        motionflow = data['mf']
        cam_t = data['cam_t']

        h, w = pixelflow.shape[0], pixelflow.shape[1]

        # Step 1: process input (noisy pixel flow and noisy depth)
        visible_pixel_y, visible_pixel_x = np.where(mask > 0)
        
        # subsample some pixel flow
        n_points = len(visible_pixel_x)
        sample_idx = np.random.permutation(n_points)
        sample_idx = sample_idx[:max([400, int(n_points * 0.005)])]#np.random.uniform(1/25, 1.0, 1)[0])])]    # we at least use 400 pixels for tracking.

        visible_pixel_x = visible_pixel_x[sample_idx]
        visible_pixel_y = visible_pixel_y[sample_idx]
        visible_pixel_locs = np.stack([visible_pixel_y, visible_pixel_x], axis=-1)

        input_pixelflow = np.zeros((h, w, 3))
        input_pixelflow[visible_pixel_y, visible_pixel_x] = pixelflow[visible_pixel_y, visible_pixel_x]

        input_pixel_mask = np.zeros((h, w, 1))
        input_pixel_mask[visible_pixel_y, visible_pixel_x] = 1.0

        input_depth = np.array(depth)
        
        if self.randomizer is not None:
            input_depth, info = self.randomizer.add_depth_noise(input_depth, mask)
            input_pixelflow, _ = self.randomizer.add_pixelflow_noise(pixelflow_map=input_pixelflow, pixel_locs=visible_pixel_locs, info=info)

        x = self.wrap_input(mask[..., None], input_depth, input_pixel_mask, input_pixelflow)
        y = self.wrap_output(depth, motionflow)

        fov = data["fovy"]
        f = h / 2 / math.tan(fov / 2) 
        cx = w // 2
        cy = h // 2

        camera_intrinsics = np.array([f, f, cx, cy])
        out = {
            "x": x.astype(np.float32),
            "cam_t": self.flatten_transform(cam_t).astype(np.float32),
            "cam": camera_intrinsics.astype(np.float32), #[fx, fy, cx, cy]
            "y": y.astype(np.float32),
            "mask": mask[None].astype(np.float32), 
        }
        return out
