# 3D Motion Field (3DMF)
Public Repository for Object-centric 3D Motion Field for Robot Learning from Human Videos

## Installation
Clone this repository and install the following packages.
- Segment Anything 2 (SAM2).
- Pytorch ``pip install torch torchvision``
- Diffusers ``pip install diffusers``
- Utilities ``pip install imagio[ffmpeg] scipy``

## Quick Demo (3D Motion Field Extraction)
Let’s start with a simple demo. In this section, we will use a pretrained model to label standard 4:3 RGBD videos captured by commonly used RGBD cameras (e.g., RealSense). The only requirement is that your video has a 4:3 aspect ratio with a vertical field of view of approximately 40–50 degrees. If your video is in 16:9 format, you will need to crop it beforehand.

Note: In the original paper, we used 256×256 images in a 1:1 format. We later trained the model on a larger dataset of standard 4:3 format (rescaled to 400×300) to eliminate the need for cropping.

### Preparation (Example Dataset and Checkpoint)
- Example Video: We provided an example video in the Release folder. Please download it and extract it to the ``./asset/video`` folder. File structure: ``./asset/video/demo/demo1_**``.

- Checkpoint: Please download the checkpoint from and extract it to the ``./asset/checkpoint`` folder. File structure: ``./asset/checkpoint/pretrained/[last_ema.pth|config.json]``.

### Run
```
python ./label_data.py --vis --path ./asset/video/demo
```
The ``--vis`` flag here will produce a mp4 visualization at ``visualize_motion.mp4``. The raw labeled motion data is at `` ./asset/video/demo_mpilabeled``.

In the visualization video, you will see something like
![](./misc/example.png)

### Label Your Own Video
To use your own video, place the following files in a folder following the structure above, then run the command:  
  - `XXX_bbox.npz`  
  - `XXX_camera.npz`  
  - `XXX_depth.npz`  
  - `XXX_rgb.mp4`  

File formats:  
  - **`bbox`**: an `[N, 4]` array representing axis-aligned bounding boxes in the format `[top-left x, top-left y, bottom-right x, bottom-right y]`.  
  - **`camera`**: a dictionary containing the intrinsic parameters `fx`, `fy`, `cx`, `cy`.  
  - **`depth`**: raw depth values; divide by `1000` to convert to metric depth in meters.  

### Note
- You may notice that the motion field in the human video fluctuates slightly over time, which is likely due to the natural physiological tremor present in everyone’s muscles (1). The human hand exhibits continuous, small-amplitude oscillations at around 7 Hz, with peak accelerations typically near 10 cm/s² and can occasionally reaching 30 cm/s².
(1) [A Normative Study of Postural Tremor of the Hand. 1982. ](https://jamanetwork.com/journals/jamaneurology/article-abstract/580802)

## Train a Motion Estimator
In this section, we demonstrate how to train the motion detector introduced above. For this version, all you need is a folder containing segmented depth images along with their corresponding camera intrinsics. Our program will automatically generate random motions and use them to train the motion estimator.
### Preparation (Example Dataset)
- Example Video: We provided an example dataset in the Release folder for a quick sanity check. Please download it and extract it to the ``./asset/dataset`` folder. File structure: ``./asset/dataset/motion_dataset``.

### Run
```
python train_motion_field.py
```

### Train with Your Own Dataset
At the moment, please refer to the ``mf_module/data/processor.py`` Line 125 (generate_data function) for data format.

## Train Motion Policy & Deploy.
Finally, we use the motion estimators described above to label human videos and train motion policies. After running ``label_data.py``, you will obtain a folder containing ``(image, motion)`` pairs. We then use this data to train a diffusion model.

### Run
(Note: Come back to this later. under construction)
```
python train_policy.py
```

## Notes
In our early version, we used GroundingDINO to automatically label bounding boxes. With the recent release of SAM3, the effort required for bounding box annotation can be significantly reduced.

