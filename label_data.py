import pickle 
import os 
import numpy as np
import torch 
import argparse 
import cv2
import imageio
import time 

from pathlib import Path 
from tqdm import tqdm 

from io import BytesIO
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mf_module.agent import get_ckpt_dir
from mf_module.agent.sam import SAM2Video
from mf_module.agent.tracker import Cotracker
from mf_module.agent.motion_detection import MotionDetectionAgent


def save_frames_to_mp4_imageio(frames, filename, fps=30):
    """
    Save a list of RGB frames (NumPy arrays) to an MP4 file using imageio.
    
    Args:
        frames (list or iterable): List of RGB frames as NumPy arrays (H x W x 3).
        filename (str): Output video file path, e.g., 'output.mp4'.
        fps (int): Frames per second.
    """
    # Create a writer object with desired fps and codec
    writer = imageio.get_writer(filename, fps=fps, codec='libx264')

    for frame in frames:
        # Make sure frames are uint8
        frame_uint8 = np.ascontiguousarray(frame.astype('uint8'))  
        writer.append_data(frame_uint8[:, :, ::-1])

    writer.close()


def save_frames_to_mp4(frames, filename, fps=30):
    """
    Save a list of RGB frames (NumPy arrays) to an MP4 file.
    
    Args:
        frames (list or iterable): List of RGB frames as NumPy arrays (H x W x 3).
        filename (str): Output video file path, e.g., 'output.mp4'.
        fps (int): Frames per second.
    """
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for f in frames:
        out.write(f.astype('uint8'))

    out.release()


def read_all_frames(video_path):
    """ Read all the video frames and return it as a list.
    """
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print("Error: Could not open video file.")
        return []
    frames = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frames.append(frame)
 
    video_capture.release()
    return frames


def reshape_depth_frames(depth_frames, shape):
    """ Reshape a list of depth frames.
    """
    result = []
    for depth in tqdm(depth_frames, desc="Reshaping Depth Frames"):
        result.append(cv2.resize(depth, shape, interpolation=cv2.INTER_LINEAR))
    return np.array(result)


def parse_video_segmentation(
    segmentation_model,
    rgb_path,
    object_label_path,
    save_path
):
    object_initial_bbox = np.load(object_label_path)['bbox']
    video_segmentation = segmentation_model.forward(rgb_path.as_posix(), {"bbox": object_initial_bbox})
    with open(save_path, "wb") as f:
        pickle.dump(video_segmentation, f)
    return video_segmentation


def parse_object_pixel_flow(
    tracker, 
    video_frames, 
    object_mask
): 
    track = tracker.forward(
        video_chunk=np.array(video_frames), 
        mask=object_mask
    )
    return track


def dump_reshaped_rgbd_video(
    rgb_source_path, 
    rgb_target_path, 
    depth_source_path, 
    depth_target_path,
    camera_info_source_path, 
    camera_info_target_path,
    object_label_source_path,
    object_label_target_path,
    target_resolution
):
    # process rgb video
    rgb_frames = read_all_frames(rgb_source_path)
    if len(rgb_frames) == 0:
        return

    rgb_frames_reshaped = [cv2.resize(img, target_resolution) for img in rgb_frames]
    save_frames_to_mp4(rgb_frames_reshaped, rgb_target_path)

    h, w, _ = rgb_frames[0].shape
    target_w, target_h = target_resolution

    # process depth video
    depth_frames = np.load(depth_source_path)['depth']
    depth_frames = reshape_depth_frames(depth_frames, target_resolution)
    np.savez_compressed(depth_target_path, depth=depth_frames)

    # process camera info
    camera_info = np.load(camera_info_source_path)
    np.savez_compressed(
        camera_info_target_path, 
        cx=camera_info["cx"] * target_w / w, 
        cy=camera_info["cy"] * target_h / h, 
        fx=camera_info["fx"] * target_w / w, 
        fy=camera_info["fy"] * target_h / h
    )

    # process bbox
    object_initial_bbox = np.load(object_label_source_path)['bbox']
    object_initial_bbox_reshaped = []

    for bbox in object_initial_bbox:
        bbox_reshaped = [bbox[0] * target_w / w, bbox[1] * target_h / h, bbox[2] * target_w / w, bbox[3] * target_h / h]
        object_initial_bbox_reshaped.append(np.array(bbox_reshaped).reshape(-1))

    object_initial_bbox_reshaped = np.array(object_initial_bbox_reshaped)
    np.savez_compressed(object_label_target_path, bbox=object_initial_bbox_reshaped)
    return


# in this function, we will label all the elements in the video.
class VideoParser:
    def __init__(
        self, 
        segmentation_model, 
        pixel_tracker,
        motion_parser,
        tracker_chunk=5
    ):
        self.segmentation_model = segmentation_model
        self.pixel_tracker = pixel_tracker
        self.motion_parser = motion_parser
        self.tracker_chunk = tracker_chunk

    def parse_segmentation(
        self,
        rgb_path,
        object_label_path,
        save_path
    ):
        return parse_video_segmentation(
            segmentation_model=self.segmentation_model, 
            rgb_path=rgb_path,
            object_label_path=object_label_path,
            save_path=save_path
        )

    def parse_object_pixel_flow(
        self,
        rgb_path,
        video_segmentation,
        object_label_path,
        save_path,
        chunk=5,
    ):
        video_frames = np.array(read_all_frames(rgb_path))
        T = video_frames.shape[0]

        for frame_id in tqdm(range(0, T - 2), desc="extracting pixel flow"):
            video_chunk = video_frames[frame_id:frame_id + chunk]
            object_masks = self.get_object_masks(video_segmentation, frame_id)

            results = {}

            for obj_id, obj_mask in object_masks.items():
                if len(np.argwhere(obj_mask > 0)) == 0:
                    continue

                object_flow = parse_object_pixel_flow(
                    self.pixel_tracker, 
                    video_chunk, 
                    obj_mask
                )
                results[str(obj_id)] = object_flow
            np.savez(save_path / f"{frame_id}", **results)

        with open(save_path / "metadata.txt", "w") as f: 
            f.write("success: 1")

        return 

    def get_motion_field(
        self,
        depth,
        next_depth,
        pixel_trajectories,
        object_masks,
        camera_intrinsics,
        frame_skip=2
    ):
        result = self.motion_parser.init_result_data({"image_shape": depth.shape})

        for obj_id, obj_mask in object_masks.items():    
            if str(obj_id) not in pixel_trajectories:
                continue

            pixel_trajectory = pixel_trajectories[str(obj_id)][[0, frame_skip]]
           

            input_data = {
                "depth": depth, 
                "next_depth": next_depth, 
                "pixel_trajectory": pixel_trajectory, 
                "mask": obj_mask,
                "camera_intrinsics": camera_intrinsics
            }
            out = self.motion_parser.forward(**input_data)
            
            if out is None:
                print("WARNING! Motion Field is none. probably the object is occluded.")
                continue

            result = self.motion_parser.add_data_to_result(obj_id, out, result, input_data)
        return result 

    def get_object_masks(self, video_segmentation, frame_id):
        return {i: v[0] for i, v in video_segmentation[frame_id].items()}

    def parse_video(
            self, 
            video_root, 
            video_name, 
            frame_skip=2, 
            visualize=False, 
            visualize_path="visualize.mp4",
            output_path=None, 
            target_resolution=(400, 300), 
            normalize_depth=True,
            normalize_depth_factor=1000.0
        ):
        if not isinstance(video_root, Path):
            video_root = Path(video_root)


        rgb_source_path = video_root / (video_name + 'rgb.mp4')
        depth_source_path = video_root / (video_name + 'depth.npz')
        camera_info_source_path = video_root / (video_name + 'camera.npz')
        object_label_source_path = video_root / (video_name + 'bbox.npz')
        
        # we need to process that.
        os.makedirs(video_root / "processed", exist_ok=True)

        target_file_id = '_'.join([str(i) for i in target_resolution])
        rgb_target_path = video_root / "processed" / (video_name + f'{target_file_id}_rgb.mp4')
        depth_target_path = video_root / "processed" / (video_name + f'{target_file_id}_depth.npz')
        camera_info_target_path = video_root / "processed" / (video_name + f'{target_file_id}_camera.npz')
        object_label_target_path = video_root / "processed" / (video_name + f'{target_file_id}_bbox.npz')
        
        dump_reshaped_rgbd_video(
            rgb_source_path, 
            rgb_target_path, 
            depth_source_path, 
            depth_target_path,
            camera_info_source_path, 
            camera_info_target_path,
            object_label_source_path,
            object_label_target_path,
            target_resolution
        )

        rgb_path = rgb_target_path
        depth_path = depth_target_path
        camera_info_path = camera_info_target_path
        object_label_path = object_label_target_path

        segmentation_path = video_root / "processed" / (video_name + f'{target_file_id}_seg.pkl')
        pixel_flow_path = video_root / "track" / (video_name + target_file_id)
        
        if not os.path.exists(segmentation_path):
            video_segmentation = self.parse_segmentation(
                rgb_path=rgb_path,
                object_label_path=object_label_path,
                save_path=segmentation_path
            )

        else:
            with open(segmentation_path, "rb") as f:
                video_segmentation = pickle.load(f)

        if not os.path.exists(pixel_flow_path / "metadata.txt"):
            os.makedirs(pixel_flow_path, exist_ok=True)

            self.parse_object_pixel_flow(
                rgb_path=rgb_path,
                object_label_path=object_label_path,
                video_segmentation=video_segmentation,
                save_path=pixel_flow_path
            )

        with torch.no_grad():
            all_video_frame_info = self._parse_video(
                rgb_path=rgb_path,
                depth_path=depth_path,
                object_label_path=object_label_path,
                pixel_flow_path=pixel_flow_path,
                camera_info_path=camera_info_path,
                video_segmentation=video_segmentation,
                frame_skip=frame_skip,
                visualize=visualize,
                visualize_path=visualize_path,
                normalize_depth=normalize_depth,
                normalize_depth_factor=normalize_depth_factor
            )

            self.process_and_save_data(all_video_frame_info, output_path)

        return 

    def reduce_all_object_masks(self, mask_dict):
        all_masks = np.array([np.squeeze(v) for _, v in mask_dict.items()])
        return np.sum(all_masks, axis=0)

    def process_and_save_data(self, all_video_frame_info, output_path=None):
        """
        Our latest organization:

        data 
        |- common
        |     |- [0~T-1].npz 
        |- motion
              |- [0~T-1].npz
    

        """
        T = len(all_video_frame_info)

        common_data_path = output_path / "common"
        motion_data_path = output_path / "motion"

        os.makedirs(common_data_path, exist_ok=True)
        os.makedirs(motion_data_path, exist_ok=True)

        for t in range(T):
            frame_info = all_video_frame_info[t]

            output_data = {
                "rgb": frame_info["rgb"],                   # [H, W, 3]
                "depth": frame_info["depth"],               # [H, W]
                "mask": frame_info["mask"]                  # [H, W]
            }

            output_motion_data = {}
            output_motion_data.update(
                self.motion_parser.convert_result_to_save_format(frame_info["motion"])
            )

            if common_data_path is not None:
                np.savez(common_data_path / f"{t}", **output_data)
                np.savez(motion_data_path / f"{t}", **output_motion_data)

            else:
                if t == 0:
                    print("Warning: Output Path is None!")

        return 

    def get_all_object_segmentation_map(self, video_segmentation, h, w):
        mask = np.zeros((h, w)).astype(np.int32)
        for obj_id, obj_mask in video_segmentation.items():
            obj_mask = obj_mask.squeeze().astype(np.int32)
            mask = mask * (1 - obj_mask) + obj_mask * obj_id 
        return mask 

    def _parse_video(
        self,
        rgb_path,
        depth_path, 
        object_label_path,
        camera_info_path,
        pixel_flow_path,
        video_segmentation,
        frame_skip=2,
        visualize=False,
        visualize_path="visualize.mp4",
        prefetched_video_frames=None,
        normalize_depth=True,
        normalize_depth_factor=1000.0
    ):
        # Load data.
        depth_frames = np.load(depth_path)['depth']
        if normalize_depth:
            depth_frames = depth_frames / normalize_depth_factor

        if prefetched_video_frames is not None:
            video_frames = prefetched_video_frames
        else:
            video_frames = np.array(read_all_frames(rgb_path))

        object_initial_labels = np.load(object_label_path)['bbox']
        camera_info = np.load(camera_info_path)

        if depth_frames.shape[1] != video_frames.shape[1] or depth_frames.shape[2] == video_frames.shape[2]:
            depth_frames = reshape_depth_frames(depth_frames, (video_frames[0].shape[1], video_frames[0].shape[0]))
        
        img_h, img_w = depth_frames.shape[1], depth_frames.shape[2]
    
        all_motion_frame = []
        all_outputs = []

        if video_segmentation is None:
            with open(segmentation_path, "rb") as f:
                video_segmentation = pickle.load(f)

        T = len(video_frames)

        for frame_id in tqdm(range(0, T - frame_skip * 2, frame_skip), "Labeling Motion Field"):
            output = {}

            pixel_trajectories = np.load(pixel_flow_path / f"{frame_id}.npz")
            object_masks = self.get_object_masks(video_segmentation, frame_id)

            motion_field_info = None 

            if self.motion_parser is not None:
                motion_field_info = self.get_motion_field(
                    depth=depth_frames[frame_id],
                    next_depth=depth_frames[frame_id+frame_skip],
                    pixel_trajectories=pixel_trajectories,
                    object_masks=object_masks,
                    camera_intrinsics=np.array([
                        camera_info["fx"], 
                        camera_info["fy"], 
                        camera_info["cx"], 
                        camera_info["cy"]
                    ]),
                    frame_skip=frame_skip
                )

            if visualize:
                if self.motion_parser is not None:
                    motion_frame = self.render_motion(
                        motion_field_info["visual"], 
                        video_frames[frame_id][:, :, ::-1],
                        self.reduce_all_object_masks(object_masks), 
                        frame_id=frame_id / frame_skip
                    )
                    motion_frame_h, motion_frame_w = motion_frame.shape[0], motion_frame.shape[1]
                    new_frame_w = int(motion_frame_w / motion_frame_h * 256)

                    motion_frame = cv2.resize(motion_frame, (new_frame_w, 256), interpolation=cv2.INTER_LINEAR)
                    all_motion_frame.append(motion_frame[:, :, ::-1])

            """
            - array
                - [{"motion: ..., "visual": ....}]
            - id
            """
            output = {
                "rgb":      video_frames[frame_id], # [H, W, 3]
                "depth":    depth_frames[frame_id], # [H, W] 
                "motion":   motion_field_info["motion"],
                "camera":   camera_info,
                "mask":     self.get_all_object_segmentation_map(video_segmentation[frame_id], img_h, img_w)
            }

            all_outputs.append(output)

        # we need to dump all the parsed information to the disk.
        if visualize:
            if self.motion_parser is not None:
                save_frames_to_mp4_imageio(all_motion_frame, visualize_path, fps=15)
        return all_outputs

    def render_motion(self, motion, rgb, mask, frame_id=0):
        motion = np.moveaxis(motion, -1, 0)

        motion_image = np.array(motion) * 0.0
        for i in range(4):
            motion_image[i] = np.ma.masked_where((1 - mask).astype(bool), np.array(motion[i]))

        # Custom colormap
        cmap = plt.cm.plasma.copy()
        cmap.set_bad(color='#F8F8F8', alpha=1.0)

        # Create figure
        fig, axes = plt.subplots(1, 5, figsize=(14, 3))
        TITLES = [f'Z_{frame_id}', 'dX', 'dY', 'dZ']
        SCALE = 0.02
        AMIN = [0.4, -SCALE, -SCALE, -SCALE]
        AMAX = [1.0, SCALE, SCALE, SCALE]

        axes[0].imshow(rgb)
        axes[0].set_title(f'RGB', fontsize=10)
        axes[0].axis('off')

        for i, ax in enumerate(axes[1:]):
            title = TITLES[i]
            im = ax.imshow(motion_image[i], cmap=cmap, vmin=AMIN[i], vmax=AMAX[i])
            ax.set_title(f'{title}', fontsize=10)
            ax.axis('off')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)

            formatter = mticker.FormatStrFormatter('%.3f')
            cbar.ax.yaxis.set_major_formatter(formatter)

        fig.tight_layout(pad=2.0)
        
        # Render to NumPy array via buffer
        canvas = FigureCanvas(fig)
        buf = BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        img_array = np.array(img)

        # Cleanup
        plt.close(fig)
        buf.close()
        return img_array 


def parse_object_segmentation(seg_model, video_path):
    segmentations = sam_predictor.forward(video_path, {"bbox": initial_bbox})
    return segmentations


def worker_fn(video_parser, video_name, video_root, output_path, visualize=True, visualize_path="visualize.mp4", frame_skip=1):
    os.makedirs(output_path, exist_ok=True)
    video_parser.parse_video(
        video_root=video_root, 
        video_name=video_name, 
        frame_skip=1, 
        visualize=visualize, 
        visualize_path=visualize_path,
        output_path=output_path
    )


def main():
    import argparse
    from mpi4py import MPI
    import math 

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        device_ids = [int(x) for x in cuda_visible.split(',') if x.strip().isdigit()]
    else:
        device_ids = list(range(torch.cuda.device_count()))

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./', type=str)
    parser.add_argument('--output_path', default='', type=str)
    parser.add_argument('--prefix', default='', type=str)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--vis_path', default="visualize.mp4")
    parser.add_argument('--checkpoint', default="pretrained")


    args = parser.parse_args()

    if args.prefix != '':
        all_folders = os.listdir(args.path)
        folder_i = []
        folder_o = []
        for f in all_folders:
            prefix = args.prefix
            if prefix[-1] == '/':
                prefix = prefix[:-1]
            if f.startswith(prefix) and 'mpilabeled' not in f:
                folder_i.append(Path(args.path) / f)
                folder_o.append(Path(args.path) / (f + '_mpilabeled'))

    else:
        video_root_path = Path(args.path)

        if args.output_path == '':
            output_path_root = Path(video_root_path.as_posix() + '_mpilabeled')
        else:
            output_path_root = Path(args.output_path)
        
        folder_i = [video_root_path]
        folder_o = [output_path_root]
 
    device_id = device_ids[rank]
    motion_detector_name = args.checkpoint
    motion_detection_agent = MotionDetectionAgent(ckpt_path=get_ckpt_dir(motion_detector_name))
    sam2_predictor = SAM2Video(device=f"cuda")
    pixel_tracker = Cotracker(device=f"cuda")

    video_parser = VideoParser(
        segmentation_model=sam2_predictor, 
        pixel_tracker=pixel_tracker,
        motion_parser=motion_detection_agent
    ) 

    all_process_target = []
    
    for data_path_i, data_path_o in zip(folder_i, folder_o):
        all_files = [file[:-7] for file in os.listdir(data_path_i) if file.endswith('.mp4')]
        for i, f in enumerate(all_files):
            output_path = data_path_o / f"{i}"
            all_process_target.append((f, data_path_i, output_path, args.vis))

    all_process_target = comm.bcast(all_process_target, root=0)
    chunk_size = max([1, int(math.ceil(len(all_process_target) / size))])
    
    if chunk_size * rank < len(all_process_target):
        process_target_subset = all_process_target[chunk_size * rank : chunk_size * (rank + 1)]
        for target in process_target_subset:
            worker_fn(video_parser, *target)
        
    return 


if __name__ == '__main__':
    main()
