import cv2
import h5py
import torch
import numpy as np
from PIL import Image
from torch.utils import data
import matplotlib.pyplot as plt
from torchvision import transforms

import os

import data.transforms as T

# TODO: IMPLEMENT SOME WAY TO GET THE NUMBER OF CLASSES

resolution_dict = {
    'full' : (480, 640),
    'half' : (240, 320),
    'mini' : (224, 224)
    }


class NYUv2Dataset(data.Dataset):
    def __init__(self, root: str, split: str, transform=None):
        self.root = os.path.join(root, split)

        for r, _, files in os.walk(self.root):
            if r.endswith('depth'):
                self.depth_paths = list(map(lambda x: os.path.join(r, x), files))
            elif r.endswith('image'):
                self.image_paths = list(map(lambda x: os.path.join(r, x), files))
            elif r.endswith('label'):
                self.label_paths = list(map(lambda x: os.path.join(r, x), files))

        self.transform = transform

        print(f'[INFO] Dataset Loaded: NYUv2 ({split})')

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = np.load(self.image_paths[index]) # RGB image (288, 384, 3)
        depth = np.load(self.depth_paths[index]) # Depth ground truth [0, 10] (288, 384, 1)
        label = np.load(self.label_paths[index]) # Segmentation map (288, 384)

        sample = {'image': image,
                  'depth': depth.reshape((288,384)),    # Transform to shape (288, 384)
                  'label': label}
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    
    def __repr__(self) -> str:
        outstr = '\n[Dataset Information]\n'
        outstr += f'Number of images: {len(self.image_paths)} \n' 
        outstr += f'Number of depth: {len(self.depth_paths)} \n'
        outstr += f'Number of segmentation: {len(self.label_paths)} \n'
        return outstr
    

def train_transformation(resolution):
    transform = transforms.Compose([
        T.Resize(resolution),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomChannelSwap(0.5),
        T.ToTensor(test=False, maxDepth=10.0)
    ])
    return transform


def val_transformation(resolution):
    transform = transforms.Compose([
        T.Resize(resolution),
        T.ToTensor(test=True, maxDepth=10.0)
    ])
    return transform


def get_NYUv2_dataset(root_path, split, resolution='full'):
    resolution = resolution_dict[resolution]

    assert split in ['train', 'val', 'test'], '[ERROR] Invalid value for: split'

    # Get transforms
    transform = train_transformation(resolution) if split == 'train' else val_transformation(resolution)

    # Load dataset corresponding to the split
    dataset = NYUv2Dataset(root_path, split, transform= transform)
    print(dataset)

    return dataset





