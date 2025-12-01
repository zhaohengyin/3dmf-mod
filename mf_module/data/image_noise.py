import cv2
import numpy as np
import random 
from scipy.signal import convolve2d


def random_soft_directional_kernels(direction_bias=None, temp=5.0):
    # Base directional templates (biased toward center + one direction)
    base_from_up_5x5 = np.array([
        [0.02, 0.05, 0.10, 0.05, 0.02],
        [0.00, 0.10, 0.30, 0.10, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00]
    ])
    base_from_up_5x5 = base_from_up_5x5 / base_from_up_5x5.sum()
    base_from_down_5x5 = np.array([
        [0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.10, 0.30, 0.10, 0.00],
        [0.02, 0.05, 0.10, 0.05, 0.02]
    ])
    base_from_down_5x5 = base_from_down_5x5 / base_from_down_5x5.sum()

    base_from_left_5x5 = np.array([
        [0.02, 0.00, 0.00, 0.00, 0.00],
        [0.05, 0.10, 0.00, 0.00, 0.00],
        [0.10, 0.30, 0.00, 0.00, 0.00],
        [0.05, 0.10, 0.00, 0.00, 0.00],
        [0.02, 0.00, 0.00, 0.00, 0.00]
    ])
    base_from_left_5x5 = base_from_left_5x5 / base_from_left_5x5.sum()

    base_from_right_5x5 = np.array([
        [0.00, 0.00, 0.00, 0.00, 0.02],
        [0.00, 0.00, 0.00, 0.10, 0.05],
        [0.00, 0.00, 0.00, 0.30, 0.10],
        [0.00, 0.00, 0.00, 0.10, 0.05],
        [0.00, 0.00, 0.00, 0.00, 0.02]
    ])
    base_from_right_5x5 = base_from_right_5x5 / base_from_right_5x5.sum()

    def randomize(kernel, noise_level=0.1):
        noise = (np.random.rand(*kernel.shape) - 0.5) * noise_level
        randomized = kernel + noise
        randomized[randomized < 0] = 0
        return randomized / randomized.sum()  # Normalize

    all_kernel = [base_from_up_5x5, base_from_down_5x5, base_from_left_5x5, base_from_right_5x5]
    if direction_bias is None:
        return random.choice(all_kernel)
    else:
        direction_bias = direction_bias / np.linalg.norm(direction_bias)
        score_up = np.exp(direction_bias[1] * temp)
        score_down = np.exp(-direction_bias[1] * temp)
        score_left = np.exp(-direction_bias[0] * temp)
        score_right = np.exp(direction_bias[0] * temp)
        prob = np.array([score_up, score_down, score_left, score_right])
        prob = prob / np.sum(prob)
        index = np.random.choice(len(prob), p=prob)

        return all_kernel[index]


def sample_bernoulli(probs):
    return (np.random.rand(*probs.shape) < probs).astype(np.uint8)


def get_boundary(mask):
    # mask: [H, W]
    mask = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    boundary = mask - eroded
    return boundary.astype(np.float32)


def expand_mask(mask, iterations=1):
    mask = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    expanded = cv2.dilate(mask, kernel, iterations=iterations)
    return expanded.astype(np.float32)


def area_to_boundary_ratio(mask):
    boundary = get_boundary(mask)
    n_b = np.sum(boundary) 
    area = np.sum(mask)

    if np.sum(boundary) == 0:
        return 0.0
    else:
        return area / n_b

def get_random_boundary(mask, rate=0.8, steps=200, direction_bias=None, cut_off=0.25):
    out_mask = mask * 0
    n_visible = np.sum(mask)

    boundary = get_boundary(mask)
    boundary_pix_y, boundary_pix_x = np.where(boundary > 0)
    len_boundary = len(boundary_pix_x)

    selection = np.random.randint(0, len_boundary - 1, int(rate * len_boundary))
    boundary_pix_x = boundary_pix_x[selection]
    boundary_pix_y = boundary_pix_y[selection]

    out_mask[boundary_pix_y, boundary_pix_x] = 1.0
    last_mask = out_mask

    for i in range(steps):
        if np.sum(out_mask) > cut_off * n_visible:
            break
        kernel = random_soft_directional_kernels(direction_bias=direction_bias)
        last_mask = sample_bernoulli(convolve2d(last_mask, kernel, mode='same'))
        out_mask = np.clip(out_mask + last_mask, 0, 1) * mask

    return out_mask


def get_random_internal(mask, rate=0.02, steps=8, direction_bias=None, cut_off=0.1):
    out_mask = mask * 0
    n_visible = np.sum(mask)

    seed_pix_y, seed_pix_x = np.where(mask > 0)
    len_seed = len(seed_pix_y)

    selection = np.random.randint(0, len_seed - 1, int(rate * len_seed))
    seed_pix_x = seed_pix_x[selection]
    seed_pix_y = seed_pix_y[selection]

    out_mask[seed_pix_y, seed_pix_x] = 1.0
    last_mask = out_mask

    for i in range(steps):
        if np.sum(out_mask) > cut_off * n_visible:
            break
        kernel = random_soft_directional_kernels(direction_bias=direction_bias)
        last_mask = sample_bernoulli(convolve2d(last_mask, kernel, mode='same'))
        out_mask = np.clip(out_mask + last_mask, 0, 1) * mask
    return out_mask


def bernoulli_pm1(p):
    # Sample from Bernoulli(p): 1 with prob=p, 0 with prob=1-p
    sample = np.random.rand(*p.shape) < p
    return 2 * sample.astype(np.float32) - 1
