from PIL import Image, ImageOps

try:
    import accimage
except ImportError:
    accimage = None
import numbers
import random
import collections
from torchvision import transforms
import math
import cv2
import numpy as np


class Compose(object):
    """Composes several transforms together returning the same transformation applied
        to both the input and the target (label).
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input, target):
        for t in self.transforms:
            input, target = t(input, target)
        return input, target


class Scale(object):
    """
    Class intented to work in cases where we want to apply same transformation to both
    the input and the target.
    It scales the given PIL.images (input and output) to the same size.
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.scale = transforms.Scale(size=size, interpolation=interpolation)

    def __call__(self, input, target):
        if target is None:
            return self.scale(input), target
        else:
            return self.scale(input), self.scale(target)


class ToNumpy(object):
    def __call__(self, input, target):
        if target is None:

            np_input = np.array(input.getdata()).astype(np.float32)
            return np_input.reshape(input.size), target
        else:
            np_input = np.array(input.getdata()).astype(np.float32)
            # np_target is loaded in format 0-1 and transforms.ToTensor()
            # expects it in format 0-255
            np_target = np.array(target.getdata()).astype(np.float32)*255

            return np_input.reshape(input.size[0], input.size[1], 3), \
                   np_target.reshape(target.size[0], target.size[1], 1)


class RandomCrop(object):
    """
    Class intented to work in cases where we want to apply same transformation to both
    the input and the target.
    It scales the input and the output to the same size.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, input, target):
        """
        Args:
            List (img (PIL.Image)): Images to be cropped.
        Returns:
            List(PIL.Image): Cropped images.
        """
        if target != None and input.size != target.size:
            raise Exception("Input and Target must have the same size")

        if self.padding > 0:
            input = ImageOps.expand(input, border=self.padding, fill=0)
            if target is not None:
                target = ImageOps.expand(target, border=self.padding, fill=0)

        w, h = input.size
        th, tw = self.size
        if w == tw and h == th:
            return (input, target)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        if target is None:
            return input.crop((x1, y1, x1 + tw, y1 + th)), target
        else:
            return input.crop((x1, y1, x1 + tw, y1 + th)), target.crop((x1, y1, x1 + tw, y1 + th))


class ToTensor(object):
    """
    Convert a set input, target ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a set input, target PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self):
        self.toTensor = transforms.ToTensor()

    def __call__(self, input, target):
        if target is None:
            return self.toTensor(input), target
        else:
            return self.toTensor(input), self.toTensor(target)


class RandomShiftScaleRotate(object):
    def __init__(self, shift_limit=(-0.0625, 0.0625),
                 scale_limit=(1 / 1.1, 1.1),
                 rotate_limit=(-45, 45), aspect_limit=(1, 1),
                 borderMode=cv2.BORDER_REFLECT_101, prob=0.5):
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.aspect_limit = aspect_limit
        self.borderMode = borderMode
        self.prob = prob

    def __call__(self, input, target):
        if random.random() < self.prob:
            height, width, channel = input.shape

            angle = random.uniform(self.rotate_limit[0], self.rotate_limit[1])  # degree
            scale = random.uniform(self.scale_limit[0], self.scale_limit[1])
            aspect = random.uniform(self.aspect_limit[0], self.aspect_limit[1])
            sx = scale * aspect / (aspect ** 0.5)
            sy = scale / (aspect ** 0.5)
            dx = round(random.uniform(self.shift_limit[0], self.shift_limit[1])
                       * width)
            dy = round(random.uniform(self.shift_limit[0], self.shift_limit[1])
                       * height)

            cc = math.cos(angle / 180 * math.pi) * (sx)
            ss = math.sin(angle / 180 * math.pi) * (sy)
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx,
                                                             height / 2 + dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)

            input = cv2.warpPerspective(input, mat, (width, height), flags=cv2.INTER_LINEAR,
                                        borderMode=self.borderMode,
                                        borderValue=(0, 0,
                                                     0,))
            target = cv2.warpPerspective(target, mat, (width, height), flags=cv2.INTER_LINEAR,
                                         borderMode=self.borderMode,
                                         borderValue=(0, 0,
                                                      0,))
            target = np.reshape(target, (target.shape[0], target.shape[1], 1))
            return input, target
        return input, target
