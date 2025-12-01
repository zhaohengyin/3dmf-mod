import os
from pathlib import Path


def get_ckpt_dir(tag):
    all_folders = ["output", "asset/checkpoint"]
    for folder in all_folders:
        if not os.path.exists(folder):
            continue
            
        ds = os.listdir(folder)
        for d in ds:
            if tag in d:
                result = Path(folder) / d 
                print("Ckpt path =", result)
                return result

