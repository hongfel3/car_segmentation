from PIL import Image, ImageOps

try:
    import accimage
except ImportError:
    accimage = None
import numbers
import random


class RandomCrop_unet(object):
    """Crop the given PIL.Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
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
        if input.size != target.size:
            raise Exception("Input and Target must have the same size")

        if self.padding > 0:
            input = ImageOps.expand(input, border=self.padding, fill=0)
            target = ImageOps.expand(target, border=self.padding, fill=0)

        w, h = input.size
        th, tw = self.size
        if w == tw and h == th:
            return (input, target)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return (input.crop((x1, y1, x1 + tw, y1 + th)), target.crop((x1, y1, x1 + tw, y1 + th)))
