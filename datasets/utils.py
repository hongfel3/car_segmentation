from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np



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
