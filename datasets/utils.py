from torchvision import transforms

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

to_PIL = transforms.ToPILImage()


def im_show(img_list):
    """

    :param img_list:
    :return:
    """

    for idx, img in enumerate(img_list):
        img = np.array(to_PIL(img))
        plt.subplot(100 + 10 * len(img_list) + (idx + 1))
        plt.imshow(img)

    plt.show()
