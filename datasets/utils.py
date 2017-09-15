import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


class ToPILImage(object):
    """Convert a tensor to PIL Image.

    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL.Image while preserving the value range.
    """

    def __call__(self, pic):
        """
        Args:
            pic (Tensor or numpy.ndarray): Image to be converted to PIL.Image.

        Returns:
            PIL.Image: Image converted to PIL.Image.

        """
        npimg = pic
        mode = None
        if isinstance(pic, torch.FloatTensor):
            pic = pic.mul(255).byte()
        if torch.is_tensor(pic):
            if len(pic.shape)==2:
                npimg = np.transpose(pic.numpy().reshape(1, pic.shape[0], pic.shape[1]), (1, 2, 0))
            elif len(pic.shape)==3:
                npimg = np.transpose(pic.numpy().reshape(pic.shape[0], pic.shape[1], pic.shape[2]), (1, 2, 0))


        assert isinstance(npimg, np.ndarray), 'pic should be Tensor or ndarray'
        if npimg.shape[2] == 1:
            npimg = npimg[:, :, 0]

            if npimg.dtype == np.uint8:
                mode = 'L'
            if npimg.dtype == np.int16:
                mode = 'I;16'
            if npimg.dtype == np.int32:
                mode = 'I'
            elif npimg.dtype == np.float32:
                mode = 'F'
        else:
            if npimg.dtype == np.uint8:
                mode = 'RGB'
        assert mode is not None, '{} is not supported'.format(npimg.dtype)
        return Image.fromarray(npimg, mode=mode)


def im_show(img_list):
    """
    It receives a list of images and plots them together
    :param img_list:
    :return:
    """
    to_PIL = ToPILImage()

    for idx, img in enumerate(img_list):

        img = np.array(to_PIL(torch.squeeze(img)))
        plt.subplot(100 + 10 * len(img_list) + (idx + 1))
        plt.imshow(img)
        plt.colorbar()

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

def mask_to_RLEstring(mask):
    return ' '.join(str(x) for x in rle_encode(mask))


def save_checkpoint(state, is_best, folder='./checkpoints/', filename='checkpoint'):
    """

    :param state:
    :param is_best:
    :param folder:
    :param filename:
    :return:
    """
    path = folder + filename + '.pth.tar'
    torch.save(state, path)
    if is_best:
        shutil.copyfile(path, folder + filename + '_best.pth.tar')


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


def to_np(x):
    """
    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/main.py#L20
    :param x:
    :return:
    """
    return x.data.cpu().numpy()
