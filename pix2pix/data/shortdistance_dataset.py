from pix2pix.data.base_dataset import BaseDataset
from pix2pix.data.image_folder import make_dataset
from pix2pix.util.guidedfilter import GuidedFilter

import numpy as np
import os
import torch
from PIL import Image
import cv2


def normalize(img):
    img = img * 2
    img = img - 1
    return img


def normalize01(img):
    return (img - torch.min(img)) / (torch.max(img) - torch.min(img))


def read_image_data(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class ShortDistanceDataset(BaseDataset):

    def __init__(self, opt, transform=None):
        BaseDataset.__init__(self, opt)
        self.dir_rgb = os.path.join(opt.dataroot, opt.phase, 'rgb')
        self.dir_depth_map = os.path.join(opt.dataroot, opt.phase, 'depth_map')

        self.rgb_paths = os.listdir(self.dir_rgb)
        self.depth_map_paths = os.listdir(self.dir_depth_map)

        self.transform = transform

    def __getitem__(self, index):
        # Read data
        rgb_path = os.path.join(self.dir_rgb, self.rgb_paths[index])
        depth_map_path = os.path.join(self.dir_depth_map, self.depth_map_paths[index])
        data_rgb = read_image_data(rgb_path)  # needs to be a tensor
        data_depth_map = read_image_data(depth_map_path)  # needs to be a tensor

        # Only take one channel for the depth map
        data_depth_map = torch.tensor(data_depth_map).float()
        data_depth_map = data_depth_map.permute((1, 2, 0)).numpy()[:, :, 0:1]

        # Transform
        if self.transform is not None:
            combined_images = self.rgb_transform(image=data_rgb, mask=data_depth_map)
            data_rgb, data_depth_map = combined_images['image'], combined_images['mask']
            data_depth_map = data_depth_map.squeeze(0)

        return data_rgb, data_depth_map

    def __len__(self):
        """Return the total number of images."""
        return len(self.rgb_paths)
