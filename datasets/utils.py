from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import shutil
import torch

def im_show(img_list):
    """
    It receives a list of images and plots them together
    :param img_list:
    :return:
    """
    to_PIL = transforms.ToPILImage()

    for idx, img in enumerate(img_list):
        img = np.array(to_PIL(img))
        plt.subplot(100 + 10 * len(img_list) + (idx + 1))
        plt.imshow(img)

    plt.show()


def rle_encode(mask_image):
    """
    receives a masked image and encodes it to RLE
    :param mask_image:
    :return:
    """
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def save_checkpoint(state, is_best, filename='./checkpoints/checkpoint.pth.tar'):
    """
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    :param state:
    :param is_best:
    :param filename:
    :return:
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class AverageMeter(object):
    """
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
