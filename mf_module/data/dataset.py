import os 
import numpy as np
from pathlib import Path 
import math 
import cv2
import random


def find_idx(A, x):
    idx = np.searchsorted(A, x, side='right') - 1
    if idx < 0:
        idx = 0

    if idx >= len(A):
        idx = len(A) - 1
    return idx


class MotionDetectionDataset:
    def __init__(self, root, data_processor, use_mult=False):
        meta_folder = root / "meta"
        packages = os.listdir(meta_folder)
        self.data_processor = data_processor
        self.n = 0
        self.root = root
        self.data_splits = [0]
        self.data_folders = []

        for package in packages:
            with open(meta_folder / package, "r") as f:
                length = int(f.read())
                self.n += length
                self.data_splits.append(self.n)
                self.data_folders.append(package[:-4])

        self.data_splits = np.array(self.data_splits)
        if use_mult:
            self.mult = max([25600000 // self.n, 1])
        else:
            self.mult = 1

    def __len__(self):
        return self.n * self.mult
    
    def get_input_channels(self):
        return self.data_processor.get_input_channels()

    def __getitem__(self, idx):
        idx = idx % self.n
        package_idx = find_idx(self.data_splits, idx)

        file_idx = idx - self.data_splits[package_idx] 
        data = np.load(self.root /  "data" / self.data_folders[package_idx] / f"{file_idx}.npz", allow_pickle=True)
        out = self.data_processor.process_dataset(data)
        if out is None:
            # handling bad data point
            return self.__getitem__(random.randint(0, self.n-1))
        return out
