import torch 
import numpy as np 
import cv2


def augment_query(mask, n_point_axis=16):
    H, W = mask.shape
    q = np.argwhere(mask > 0)
    xmin = np.min(q[:, 1])
    xmax = np.max(q[:, 1])

    WINDOW_SCALE = 1.0
    xw = xmax - xmin
    xc = (xmax + xmin) // 2
    sample_xmin = xc - (xw * WINDOW_SCALE) // 2
    sample_xmax = xc + (xw * WINDOW_SCALE) // 2
 
    sample_xmin = max([0, sample_xmin])
    sample_xmax = min([W - 1, sample_xmax])

    ymin = np.min(q[:, 0])
    ymax = np.max(q[:, 0])

    yw = ymax - ymin
    yc = (ymax + ymin) // 2

    sample_ymin = yc - (yw * WINDOW_SCALE) // 2
    sample_ymax = yc + (yw * WINDOW_SCALE) // 2
    
    sample_ymin = max([0, sample_ymin])
    sample_ymax = min([H - 1, sample_ymax])

    yx_min = np.array([sample_ymin, sample_xmin])
    yx_max = np.array([sample_ymax, sample_xmax])

    y_points = np.linspace(yx_min[0], yx_max[0], n_point_axis)
    x_points = np.linspace(yx_min[1], yx_max[1], n_point_axis)
    yy, xx = np.meshgrid(y_points, x_points)

    # Stack them together to form 2D coordinates
    grid = np.stack([yy, xx], axis=-1)
    q = grid.reshape(-1, 2)

    q = np.concatenate([np.zeros((q.shape[0], 1)), q[:, 1:2], q[:, 0:1]], axis=-1).astype(np.float32)
    return q


class Cotracker:
    def __init__(self, device='cuda'):
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        model = model.to(device)
        self.device = device
        self.model = model 

    def forward(self, video_chunk, mask, crop_w=128):
        q = np.argwhere(mask > 0)
        xmin = np.min(q[:, 1])
        xmax = np.max(q[:, 1])

        ymin = np.min(q[:, 0])
        ymax = np.max(q[:, 0])

        crop_center_x = (xmin + xmax) // 2
        crop_center_y = (ymin + ymax) // 2

        begin_x = max([0, crop_center_x - crop_w])
        begin_y = max([0, crop_center_y - crop_w])
        
        end_x = begin_x + crop_w * 2
        end_y = begin_y + crop_w * 2

        video_chunk = video_chunk[:, begin_y:end_y, begin_x:end_x, :]
        mask = mask[begin_y:end_y, begin_x:end_x]
        result = self._forward(video_chunk, mask)
        
        result[:, :, 0] += begin_x
        result[:, :, 1] += begin_y 
        return result


    def _forward(self, video_chunk, mask):
        '''
            video_chunk: [T, H, W, C].
            mask: [H, W]

            return: a pytorch tensor, [T, N, 2] (x,y) format
        '''
        # Define kernel (structuring element), e.g., 3x3 square
        kernel = np.ones((3, 3), np.uint8)

        # Apply erosion (the boundary pixels are usually noisy)
        mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        q = np.argwhere(mask > 0)
        perm = np.random.permutation(len(q))
        q = q[perm]
        if len(q) > 1024:
            q = q[:1024]

        q = np.concatenate([np.zeros((q.shape[0], 1)), q[:, 1:2], q[:, 0:1]], axis=-1).astype(np.float32)
        n_main_query = q.shape[0]

        q_aug = augment_query(mask)
        q = np.concatenate((q, q_aug), axis=0)
        video_chunk = torch.from_numpy(np.array(video_chunk)).permute(0, 3, 1, 2)[None, ...].float().to(self.device)

        result = self.model.forward(video_chunk, queries=torch.from_numpy(q)[None, ...].to(self.device))[0] # [T, N, 2]
        result = result.detach().cpu().numpy()[0, :, :n_main_query, :]

        return result 