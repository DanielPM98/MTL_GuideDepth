import numpy as np
import torch
from torchvision import transforms, utils
from PIL import Image
from io import BytesIO
import random

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def _is_ndarray(img):
    return isinstance(img, np.ndarray)



class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        check_is_pil(sample)
        image, depth, label = unpack(sample)

        if random.random() < self.p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'depth': depth, 'label': label}



class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        check_is_pil(sample)
        image, depth, label = unpack(sample)

        if random.random() < self.p:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
            label = label.transpose(Image.FLIP_TOP_BOTTOM)

        return {'image': image, 'depth': depth, 'label': label}


class RandomChannelSwap(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        check_is_pil(sample)
        image, depth, label = unpack(sample)
        
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[...,list(self.indices[random.randint(0, len(self.indices) - 1)])])
        return {'image': image, 'depth': depth, 'label': label}



class ToTensor(object):
    def __init__(self, test=False, maxDepth=1000.0):
        self.test = test
        self.maxDepth = maxDepth

    def __call__(self, sample):
        image, depth, label = unpack(sample)
        transformation = transforms.ToTensor()

        if self.test:
            """
            If test, move image to [0,1] and depth to [0, 10]
            """
            image = np.array(image).astype(np.float32) / 255.0
            depth = np.array(depth).astype(np.float32)
            label = np.array(label).astype(np.float32)
            image, depth, label = transformation(image), transformation(depth), transformation(label)
        else:
            #Fix for PLI=8.3.0
            image = np.array(image).astype(np.float32) / 255.0
            depth = np.array(depth).astype(np.float32)
            label = np.array(label).astype(np.float32)


            #For train use DepthNorm 
            zero_mask = depth == 0.0
            image, depth, label = transformation(image), transformation(depth), transformation(label)

            # print('Depth before, min: {} max: {}'.format(depth.min(), depth.max()))
            depth = torch.clamp(depth, self.maxDepth/100.0, self.maxDepth)

            depth = self.maxDepth / depth # Adjust for distance parameters TODO: EXPLAIN CORRECTLY
            depth[:, zero_mask] = 0.0

        # print('Depth after, min: {} max: {}'.format(depth.min(), depth.max()))
        # print('Image, min: {} max: {}'.format(image.min(), image.max()))

        image = torch.clamp(image, 0.0, 1.0)
        return {'image': image, 'depth': depth, 'label': label}



class CenterCrop(object):
    """
    Wrap torch's CenterCrop
    """
    def __init__(self, output_resolution):
        self.crop = transforms.CenterCrop(output_resolution)

    def __call__(self, sample):
        image, depth, label = unpack_ndarray(sample)

        image = self.crop(image)
        depth = self.crop(depth)
        label = self.crop(label)

        return {'image': image, 'depth': depth, 'label': label}



class Resize(object):
    """
    Wrap torch's Resize
    """
    def __init__(self, output_resolution):
        self.resize = transforms.Resize(output_resolution)

    def __call__(self, sample):
        image, depth, label = unpack_ndarray(sample)

        image = self.resize(image)
        depth = self.resize(depth)
        label = self.resize(label)

        return {'image': image, 'depth': depth, 'label': label}



class RandomRotation(object):
    """
    Wrap torch's Random Rotation
    """
    def __init__(self, degrees):
        self.angle = degrees

    def __call__(self, sample):
        image, depth, label = unpack_ndarray(sample)
        angle = random.uniform(-self.angle, self.angle)

        image = transforms.functional.rotate(image, angle)
        depth = transforms.functional.rotate(depth, angle)
        label = transforms.functional.rotate(label, angle)

        return {'image': image, 'depth': depth, 'label': label}


def unpack(sample):
    image, depth, label = sample['image'], sample['depth'], sample['label']

    return image, depth, label


def unpack_ndarray(sample):
    image, depth, label = unpack(sample)
    if _is_ndarray(image):
        image = Image.fromarray(np.uint8(image))
    if  _is_ndarray(depth):
        depth = Image.fromarray(depth)
    if  _is_ndarray(label):
        label = Image.fromarray(label)
    
    return image, depth, label

        
def check_is_pil(sample):
    image,depth, label = unpack(sample)
    if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
    if not _is_pil_image(depth):
        raise TypeError(
            'img should be PIL Image. Got {}'.format(type(depth)))
    if not _is_pil_image(label):
        raise TypeError(
            'img should be PIL Image. Got {}'.format(type(label)))
    

def DepthNorm(depth, maxDepth=1000.0):
    return maxDepth / depth
