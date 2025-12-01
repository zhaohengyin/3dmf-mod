from pathlib import Path 
import torch 
import numpy as np
import cv2
import os 
import shutil 
from pathlib import Path 
from mf_module.data import MotionDetectionParserBuilder
from mf_module.model import MotionDetectionModelBuilder
from mf_module.utils.json_utils import load_json 


class MotionDetectionAgent:
    """ This agent detects motion in a video.
    """
    def __init__(self, ckpt_path, ckpt_name="last_ema"):
        config_file = Path(ckpt_path) / "config.json"
        model_file = Path(ckpt_path) / f"{ckpt_name}.pth"

        self.config_file = config_file
        self.model_file = model_file
        config = load_json(config_file)

        model_builder = MotionDetectionModelBuilder()
        parser_builder = MotionDetectionParserBuilder()

        self.parser = parser_builder.build(config["dataset"])
        self.model = model_builder.build(config["model"])

        self.model.load_state_dict(torch.load(model_file))
        self.model = self.model.cuda()
        self.model.eval()


    def forward(self, depth, next_depth, pixel_trajectory, mask, camera_intrinsics, camera_transform=None):
        """
       
        Args: 
            depth:              [h, w], np.ndarray (float32)
            pixel_trajectory:   [2, n_points, 2], np.ndarray (float32). format: (x,y)
            mask:               [h, w], np.ndarray (float32)
        
        Returns:
            out, a dict.
            'motion'            -- 'the motion representation. used for prediction'
            'dense_motion'      -- 'the visualizable motion'
        """

        data = {
            "depth": depth,
            "next_depth": next_depth,
            "mask": mask,
            "pt": pixel_trajectory,
            "camera_intrinsics": camera_intrinsics
        }

        if camera_transform is not None:
            data['camera_transform'] = camera_transform
        
        parsed_data = self.parser.parse_realworld_data(data)    

        if parsed_data is None:
            return None 
            
        x = parsed_data["x"]
        cam = torch.from_numpy(parsed_data["cam"]).cuda().unsqueeze(0)
        cam_t = torch.from_numpy(parsed_data["cam_t"]).cuda().unsqueeze(0)
        mask = torch.from_numpy(mask).cuda().squeeze()[None, None]
        pixmask = torch.from_numpy(parsed_data["pixmask"]).cuda().unsqueeze(0)

        if isinstance(x, dict):
            for k, v in x.items():
                x[k] = torch.from_numpy(v).float().cuda().unsqueeze(0)
        else:
            x = torch.from_numpy(x).cuda().unsqueeze(0)

        out = self.model.get_motion_field(x, cam, pixmask, camera_transform=cam_t)
        out = self.parser.parse_representation_output(out, data)
        return out


    def init_result_data(self, args):
        return self.model.init_result_data(args)


    def add_data_to_result(self, obj_id, out, result, input_data):
        return self.model.add_data_to_result(obj_id, out, result, input_data, self.parser)


    def convert_result_to_save_format(self, result):
        return self.model.convert_result_to_save_format(result)
