import os
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sam2
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2Video:
    def __init__(self, device='cuda'):
        sam_root = (Path(sam2.__file__) / ".." / "..").resolve()
        sam2_checkpoint = sam_root / "./checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)        
        return
    
    def forward(self, video_path, init_info):
        bbox = init_info["bbox"]
        if len(bbox.shape) == 1:
            bbox = bbox[None, ...]

        num_objects = bbox.shape[0] 
        inference_state = self.video_predictor.init_state(video_path=video_path, offload_video_to_cpu=True)
        for i, obj_bbox in enumerate(bbox):
            _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=i,
                box=obj_bbox,
            )

        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }        

        return video_segments
