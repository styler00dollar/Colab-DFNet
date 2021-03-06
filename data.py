import os

import cv2
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import torch
from torch.utils.data import Dataset


class DS(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        for root, _, fnames in sorted(os.walk(root)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                self.samples.append(path)
        if len(self.samples) == 0:
            raise RuntimeError("Found 0 files in subfolders of: " + root)

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_path = self.samples[index]
        sample = Image.open(sample_path).convert('RGB')

        if self.transform:
            sample = self.transform(sample)

        mask = DS.random_mask()
        mask = torch.from_numpy(mask)

        return sample, mask

    @staticmethod
    def random_mask(height=256, width=256,
                    min_stroke=1, max_stroke=4,
                    min_vertex=1, max_vertex=12,
                    min_brush_width_divisor=16, max_brush_width_divisor=10):
        mask = np.ones((height, width))

        min_brush_width = height // min_brush_width_divisor
        max_brush_width = height // max_brush_width_divisor
        max_angle = 2*np.pi
        num_stroke = np.random.randint(min_stroke, max_stroke+1)
        average_length = np.sqrt(height*height + width*width) / 8

        for _ in range(num_stroke):
            num_vertex = np.random.randint(min_vertex, max_vertex+1)
            start_x = np.random.randint(width)
            start_y = np.random.randint(height)

            for _ in range(num_vertex):
                angle = np.random.uniform(max_angle)
                length = np.clip(np.random.normal(average_length, average_length//2), 0, 2*average_length)
                brush_width = np.random.randint(min_brush_width, max_brush_width+1)
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)

                cv2.line(mask, (start_y, start_x), (end_y, end_x), 0., brush_width)

                start_x, start_y = end_x, end_y
        if np.random.random() < 0.5:
            mask = np.fliplr(mask)
        if np.random.random() < 0.5:
            mask = np.flipud(mask)
        return mask.reshape((1,)+mask.shape).astype(np.float32)
