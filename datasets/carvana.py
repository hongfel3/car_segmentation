from torch.utils.data.dataset import Dataset
import torch
import os
from os.path import isfile, join

from tqdm import tqdm

from PIL import Image


class CARVANA(Dataset):
    """
        CARVANA dataset that contains car images as .jpg. Each car has 16 images
        taken in different angles and a unique id: id_01.jpg, id_02.jpg, ..., id_16.jpg
        The labels are provided as a .gif image that contains the manually cutout mask
        for each training image
    """

    def __init__(self, root, subset="train", transform=None):
        """

        :param root: it has to be a path to the folder that contains the dataset folders
        :param train: boolean true if you want the train set false for the test one
        :param transform: transform the images and labels
        """

        # initialize variables
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.subset = subset
        self.data_path, self.labels_path = [], []

        def load_images(path):
            """
            returns all the sorted image paths.

            :param path:
            :return: array with all the paths to the images
            """
            images_dir = [join(path, f) for f in os.listdir(path) if isfile(join(path, f))]
            images_dir.sort()

            return images_dir

        # load the data regarding the subset
        if self.subset == "train":
            self.data_path = load_images(self.root + "/train")
            self.labels_path = load_images(self.root + "/train_masks")
        elif self.subset == "val":
            self.data_path = load_images(self.root + "/val")
            self.labels_path = load_images(self.root + "/val_masks")
        elif self.subset == "test":
            self.data_path = load_images(self.root + "/test")
            self.labels_path = None
        else:
            raise RuntimeError('Invalid subset ' + self.subset + ', it must be one of:'
                                                                 ' \'train\', \'val\' or \'test\'')

    def __getitem__(self, index):
        """

        :param index:
        :return: tuple (img, target) with the input data and its label
        """

        # load image and labels
        img = Image.open(self.data_path[index])
        target = Image.open(self.labels_path[index]) if not self.subset == 'test' else None

        # apply transforms to both
        if self.transform is not None:
            img = self.transform(img)
            target = self.transform(target)

        return img, target

    def __len__(self):
        return len(self.data_path)
