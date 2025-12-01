import random
import math
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from mf_module.data.image_noise import *


def np_loguniform(a, b, size=None):
    return np.exp(np.random.uniform(np.log(a), np.log(b), size=size))


def with_prob(p):
    return np.random.uniform(0, 1, 1)[0] < p


def erase_circle_in_mask(mask, ratio=-1, mask_ref=None):
    H, W = mask.shape
    mask = mask.astype(np.float32)
    mask_area = np.sum(mask > 0)

    if ratio < 0:
        max_circle_area = random.randint(3, 10) / 100.0 * mask_area
    else:
        max_circle_area = ratio * mask_area

    max_radius = math.sqrt(max_circle_area / np.pi)

    if mask_ref is None:
        mask_ref = mask 

    non_zero_points = np.argwhere(mask_ref >= 0.99)
    if len(non_zero_points) == 0:
        return mask, mask * 0, (0, 0), 0

    random_point = random.choice(non_zero_points)
    y, x = random_point
    max_radius = max([1, int(max_radius)])
    mask_with_circle = mask.copy()
    cv2.circle(mask_with_circle, (x, y), max_radius, (0), thickness=-1)
    deleted_part = mask - mask_with_circle
    return mask_with_circle, deleted_part, (x, y), max_radius


def wave_distortion(mask, amplitude=3, frequency=0.05):
    H, W = mask.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    
    amplitude = np.random.uniform(amplitude / 2, amplitude, 1)[0]
    if np.random.rand() > 0.5:
        distortion = amplitude * np.sin(frequency * x)
        y_distorted = np.clip(y + distortion, 0, H-1).astype(np.float32)
        x_distorted = x.astype(np.float32)
    else:
        distortion = amplitude * np.sin(frequency * y)
        x_distorted = np.clip(x + distortion, 0, W-1).astype(np.float32)
        y_distorted = y.astype(np.float32)
    
    stretched_mask = cv2.remap(mask.astype(np.uint8), x_distorted, y_distorted, cv2.INTER_NEAREST)
    return stretched_mask


def create_random_corruption_map(mask):
    H, W = mask.shape
    mask = mask.astype(np.float32)
    out = np.zeros((H, W)).astype(np.float32)

    if random.randint(0, 99) < 30:
        mask_boundary = get_mask_boundary(mask)
        n_iter = random.randint(0, 15)
        for i in range(n_iter):
            _, deleted, _, _ = erase_circle_in_mask(mask, np.random.uniform(0.05, 0.1, 1)[0], mask_ref=mask_boundary)
            if random.randint(0, 1) == 0:
                out += (- deleted.astype(np.float32) * np.random.uniform(0.03, 0.1, 1)[0])
            else:
                out += deleted.astype(np.float32) * np.random.uniform(0.03, 0.1, 1)[0] 

    else:
        n_iter = random.randint(0, 15)
        for i in range(n_iter):
            _, deleted, _, _ = erase_circle_in_mask(mask, np.random.uniform(0.01, 0.03, 1)[0])
            if random.randint(0, 1) == 0:
                out += (-deleted.astype(np.float32) * np.random.uniform(0.03, 0.1, 1)[0])
            else:
                out += deleted.astype(np.float32) * np.random.uniform(0.03, 0.1, 1)[0] 
    return out 


def random_bipolar_uniform(size, low=0.05, high=2.0):
    mask = np.random.rand(*size) < 0.5
    noise = np.empty(size)
    noise[mask] = np.random.uniform(-high, -low, size=np.sum(mask))
    noise[~mask] = np.random.uniform(low, high, size=np.sum(~mask))
    return noise


def get_spatial_correlated_noise_v2(shape, density, sigma=3, denormalize=True):
    h, w = shape[0], shape[1]
    out = np.zeros(shape).astype(np.float32)
    if int(h*w*density) <= 0:
        return out

    pos = (np.random.uniform(0, 0.999, (int(h*w*density), 2)) * np.array([[h, w]])).astype(np.int32)
    y, x = pos[:, 0], pos[:, 1]
    out[y, x] = np.random.uniform(-1, 1, len(pos))

    correlated_noise = gaussian_filter(out, sigma=sigma)

    std = np.std(correlated_noise[y, x])
    if denormalize:
        return correlated_noise / std
    return correlated_noise


class MotionDataRandomizer:
    """
        This class randomizes the rendered data. 
    """

    def __init__(self):
        self.depth_advanced_noise_probabilty = 0.4
        self.depth_white_noise_scale = 0.0002
        self.depth_correlated_noise_scale = 0.002

        self.pixelflow_noise_xy_scale = 0.3
        self.pixelflow_noise_z_scale = 0.0002
        self.pixelflow_correlated_noise_xy_scale = 2.0
        self.pixelflow_correlated_noise_z_scale = 0.002
        self.pixelflow_correlated_noise_z_scale_perturb = 0.01

        self.debug = False 
        return 

    def get_mask_region(self, source_mask):
        """
            This function returns a bunch of masks (boundary masks) given the source mask.
        
        """
        source_mask_expanded = wave_distortion(source_mask, amplitude=5).astype(bool)
        source_mask_expanded = wave_distortion(source_mask_expanded, amplitude=3).astype(bool)

        border_mask = np.logical_xor(source_mask_expanded, source_mask).astype(np.float32)
        non_border_mask = 1 - border_mask
        return {"border": border_mask, "internal": source_mask * non_border_mask}
    
    def get_mask_holes(self, mask, n_hole):
        out_mask = mask.copy()
        deleted_parts = []
        for i in range(n_hole):
            out_mask, deleted_part, _, _ = erase_circle_in_mask(out_mask)
            deleted_parts.append(deleted_part)
        return deleted_parts, out_mask

    def add_depth_white_noise(self, depth, depth_mask):
        random_depth_white_noise = np.exp(np.random.uniform(-2, 0.5, 1)[0] * np.log(10))  

        depth = depth + np.random.normal(
            loc=0.0, 
            scale=random_depth_white_noise * self.depth_white_noise_scale, 
            size=depth.shape
        ) * depth_mask[..., None]

        return depth

    def add_depth_correlated_noise(self, depth, depth_mask):
        """
            Args:
            depth:      (H, W, 1) numpy.ndarray.
            depth_mask: (H, W) numpy.ndarray

            Return:
            noised_depth:   randomized depth.
        """
        density = np.random.uniform(0.1, 0.3, 1)[0]
        sigma   = np.random.uniform(2, 8, 1)[0]
        intensity = np_loguniform(0.01, 1.0, 1)[0] * self.depth_correlated_noise_scale
        noise_depth = get_spatial_correlated_noise_v2(
            (depth.shape[0], depth.shape[1]), 
            density=density, 
            sigma=sigma, 
            denormalize=True
        ) * depth_mask * intensity

        depth = depth + noise_depth[..., None]
        return depth

    def add_depth_noise(self, depth, depth_mask):
        """
            This function performs depth data augmentation.

            Args:
            depth:      (H, W, 1) numpy.ndarray.
            depth_mask: (H, W) numpy.ndarray

            Return:
            noised_depth:   randomized depth.
        """

        depth = depth.copy()
        # White Noise
        depth = self.add_depth_white_noise(depth, depth_mask)
        depth = self.add_depth_correlated_noise(depth, depth_mask)
       
        # With some probability, directly return.
        if with_prob(1 - self.depth_advanced_noise_probabilty) and not self.debug:
            hole = None
            return depth, {"mask": depth_mask}

        # Compute Different Mask Regions
        mask_parts = self.get_mask_region(depth_mask)
        border_mask = mask_parts["border"]
        internal_mask = mask_parts["internal"]

        # With 20% probability, we remove the border.
        if with_prob(0.5) or self.debug:
            depth = depth * internal_mask[..., None] 
            # a large shift at the border!
            if with_prob(0.5):
                random_border_depth = np.random.normal(loc=0, scale=0.03, size=depth.shape) + np.random.uniform(0.3, 3.0, 1)[0]
                depth = depth + random_border_depth * border_mask[..., None] 

        # With some probability, we create some holes on the depth map.
        if with_prob(0.50) or self.debug:
            n_hole = random.randint(1, 8)
            removed_masks, holed_mask = self.get_mask_holes(internal_mask, n_hole)
            filler = np.zeros((depth.shape[0], depth.shape[1], 1))
            for removed_mask in removed_masks:
                if with_prob(0.3):
                    noise = depth + 0.3 * get_spatial_correlated_noise_v2(
                        depth.shape[:2], 
                        density=np.random.uniform(0.3, 1.0, 1),
                        denormalize=True
                    )[..., None]

                    filler = filler + removed_mask[..., None] * noise 

            depth = depth * (1 - internal_mask[..., None]) + internal_mask[..., None] * (depth * holed_mask[..., None] + filler * (1 - holed_mask[..., None]))

        info = {
            "mask": depth_mask
        }
        return depth, info


    def add_pixelflow_noise(self, pixelflow_map, pixel_locs, info):
        """
            pixelflow_map: [H, W, 3] numpy.ndarray.
            pixel_locs:    [N, 2]    numpy.ndarray (np.int32). pixel locations. using order y, x
        """
        y, x = pixel_locs[:, 0], pixel_locs[:, 1]
        h, w = pixelflow_map.shape[0], pixelflow_map.shape[1]

        mask = np.zeros((h, w))
        mask[y, x] = 1.0

        # Correlated Noise.
        intensity_xy = np_loguniform(0.01, 1.0, 1)[0] * self.pixelflow_correlated_noise_xy_scale
        intensity_z0 = np_loguniform(0.01, 1.0, 1)[0] * self.pixelflow_correlated_noise_z_scale
        intensity_z1 = np_loguniform(0.01, 1.0, 1)[0] * self.pixelflow_correlated_noise_z_scale_perturb

        # xy
        density = 1 / len(pixel_locs)
        density_xy = np.random.uniform(0.5 * density, 2 * density, 1)[0]
        sigma = np.random.uniform(0.3, 2.5, 1)[0]
        noise_x = get_spatial_correlated_noise_v2((h, w), density=density_xy, sigma=sigma, denormalize=True) * mask * intensity_xy
        noise_y = get_spatial_correlated_noise_v2((h, w), density=density_xy, sigma=sigma, denormalize=True) * mask * intensity_xy

        # z_spatial0: dense but low-amplitude spatially correlated noise
        density_z = np.random.uniform(0.1, 0.3, 1)[0]
        sigma_z = np.random.uniform(3, 8, 1)[0]
        noise_z0 = get_spatial_correlated_noise_v2((h, w), density=density_z, sigma=sigma_z, denormalize=True) * mask * intensity_z0

        # z_spatial1: sparse but high-intensity spatially correlated noise
        density_z = np.random.uniform(0.5 * density, 10 * density, 1)[0]
        sigma_z = np_loguniform(0.3, 4, 1)[0]
        noise_z1 = get_spatial_correlated_noise_v2((h, w), density=density_z, sigma=sigma_z, denormalize=True) * mask * intensity_z1

        pixelflow_map[y, x, 0] = pixelflow_map[y, x, 0] + noise_x[y, x]
        pixelflow_map[y, x, 1] = pixelflow_map[y, x, 1] + noise_y[y, x]
        pixelflow_map[y, x, 2] = pixelflow_map[y, x, 2] + noise_z0[y, x] + noise_z1[y, x]

        # White noise.
        noise_xy = np.random.normal(loc=0, scale=np_loguniform(0.01, 1.0, 1)[0], size=(pixel_locs.shape[0], 2))
        pixelflow_map[y, x, :2] = pixelflow_map[y, x, :2] + noise_xy * self.pixelflow_noise_xy_scale
        
        noise = np.random.normal(loc=0, scale=np_loguniform(0.01, 1.0, 1)[0], size=pixel_locs.shape[0])
        pixelflow_map[y, x, 2] = pixelflow_map[y, x, 2] + noise * self.pixelflow_noise_z_scale

        # Extreme Perturbation
        perturb_idx = np.random.randint(0, len(pixel_locs), max([1, int(np_loguniform(0.01, 1, 1)[0] * 0.05 * len(pixel_locs))]))
        n_perturb = len(perturb_idx)
        perturb_val = np.random.normal(
            loc=np.random.uniform(-0.1, 0.1, 1)[0], 
            scale=np_loguniform(0.01, 1.0, 1)[0] * 0.05,
            size=(n_perturb, 2)
        )
        perturb_y, perturb_x = pixel_locs[perturb_idx, 0], pixel_locs[perturb_idx, 1]
        pixelflow_map[perturb_y, perturb_x, :2] = pixelflow_map[perturb_y, perturb_x, :2] + perturb_val

        # Boundary noise
        a2b_ratio = area_to_boundary_ratio(mask)

        boundary_noise_sign = bernoulli_pm1(
            np.ones(pixel_locs.shape[0]) * random.choice([
                np.random.uniform(0, 0.1, 1)[0], 
                np.random.uniform(0.9, 1.0, 1)[0]
            ])
        )   # [N, 2]
        boundary_noise_value = np.random.normal(loc=0, scale=np.exp(np.random.uniform(-1, 0, 1)[0] * np.log(10)), size=pixel_locs.shape[0]) 
        boundary_noise = boundary_noise_sign * boundary_noise_value
        
        z_noise_mask = get_random_boundary(
            mask, 
            rate=np.random.uniform(0.05, 0.5, 1)[0], 
            direction_bias=np.random.randn(2), 
            steps=min([random.randint(1, 5), int(a2b_ratio * 0.1)]), 
            cut_off=0.05
        )
        pixelflow_map[y, x, 2] += boundary_noise * z_noise_mask[y, x] # very large noises.

        # Bipolar randomization
        bipolar_perturb_indices = np.where(np.random.rand(pixel_locs.shape[0]) < np.exp(np.random.uniform(-2, -1, 1)[0] * np.log(10)))
        by, bx = y[bipolar_perturb_indices], x[bipolar_perturb_indices]
        pixelflow_map[by, bx, 2] = pixelflow_map[by, bx, 2] + random_bipolar_uniform(pixelflow_map[by, bx, 2].shape)

        info = {}
        return pixelflow_map, info
