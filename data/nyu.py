import cv2
import h5py
import torch
import numpy as np
from PIL import Image
from torch.utils import data
import matplotlib.pyplot as plt
from torchvision import transforms


import data.transforms as T


resolution_dict = {
    'full' : (480, 640),
    'half' : (240, 320),
    'mini' : (224, 224)
    }

class Nyuv2Dataset(data.Dataset):
    def __init__(self, 
                 root,
                 top_labels:list = None, 
                 transform=None, 
                 normalize=False, 
                 depth_norm=10.0):
        """
            Description:
                root <str>: path to the NYU dataset file (.mat)
        """
        self.top_labels = top_labels
        self.transform = transform

        # image normalization parameters
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean, std) if normalize else None

        # Train depth in range [0, 10]
        self.depth_norm = depth_norm

        # Open .mat as an h5 object
        self.h5_obj = h5py.File(root, mode='r')

        # Load desired data
        self.images = self.h5_obj['images'] # RGB images
        self.depths = self.h5_obj['depths'] # depths [0, 10]
        self.labels = self.h5_obj['labels'] # Semantic class mask for each image
        self.names = self.h5_obj['names'] # Semantic class names
        # self.instances = self.h5_obj['instances'] # instances
        # self.namesToIds = self.h5_obj['namesToIds']

        print('[INFO] Dataset Loaded: NYU')

    def add_transform(self, new_transform):
        self.transform = new_transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = cv2.rotate(self.images[index].transpose(1,2,0), cv2.ROTATE_90_CLOCKWISE) # rgb image
        image = Image.fromarray(image, mode='RGB')
        depth = cv2.rotate(self.depths[index], cv2.ROTATE_90_CLOCKWISE) # depth map
        label = cv2.rotate(self.labels[index], cv2.ROTATE_90_CLOCKWISE).astype(np.float32) # semantic segmentation

        # reduce number of labels by placing them in uncategorized class
        if self.top_labels:
            for lbl in np.unique(label).astype(int):
                if lbl not in self.top_labels:
                    label[label == lbl] = 0

        sample = {
            'image': image,
            'depth': depth,
            'label': label
        }

        # Apply the transformations
        if self.transform:
            sample = self.transform(sample)

        if self.normalize:
            sample['image'] = self.normalize(sample['image'])
            sample['depth'] = sample['depth'] / depth

        return sample  # returns dict with image, depth and label (segmentation)    


def train_transformation(resolution):
    transform = transforms.Compose([
        T.Resize(resolution),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomChannelSwap(0.5),
        T.ToTensor(test=False, maxDepth=10.0)
    ])
    return transforms


def val_transformation(resolution):
    transform = transforms.Compose([
        T.Resize(resolution),
        T.ToTensor(test=True, maxDepth=10.0)
    ])
    return transform


def get_NYUv2_dataset(root_path, split, resolution='full'):
    resolution = resolution_dict[resolution]
    generator = torch.Generator().manual_seed(42)

    # Get transforms
    train_transform = train_transformation(resolution)
    val_transform = val_transformation(resolution)
    test_transform = transforms.ToTensor()

    # Load dataset and split train/val/test splits
    dataset = Nyuv2Dataset(root_path)
    train_dataset, val_dataset, test_dataset = data.random_split(dataset, [0.64, 0.16, 0.2], generator)

    # Add transformations
    train_dataset.add_transform(train_transform)
    val_dataset.add_transform(val_transform)
    test_dataset.add_transform(test_transform)

    # Return dataloader
    if split == 'train':
        return train_dataset
    elif split == 'val':
        return val_dataset
    elif split == 'test':
        return test_dataset
    
    print('[ERROR] Invalid value for: split')
    exit(0)





