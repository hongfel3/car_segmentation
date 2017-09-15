import argparse
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

torch.backends.cudnn.benchmark = True  # enables cudnn's auto-tuner

from tqdm import tqdm

import datasets.utils as data_utils

import datasets as dsets
from datasets import transforms_unet as unet_transforms

torch.backends.cudnn.benchmark = True  # enables cudnn's auto-tuner
import models

# define the parser arguments
parser = argparse.ArgumentParser(description='Car-segmentation kaggle competition')

parser.add_argument('--dir', default="./data/", type=str, metavar='D',
                    help='path to the dataset root folder (default: ./data/')
parser.add_argument('--submit_dir', default='./submit/', type=str, metavar='L',
                    help='path to the folder to store the submit file')
parser.add_argument('--arch', '-a', metavar='ARCH', default='UNET_256_ShiftScaleRotate_noweightedLoss',
                    help='name of the architecture used, also used to store the submit file')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')

parser.add_argument('-b', '--batch-size', default=10, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--resume', default='checkpoints/UNET_256_ShiftScaleRotate_noweightedLoss_best.pth.tar', type=str,
                    metavar='PATH', help='path to latest checkpoint (or best)')

global args
args = parser.parse_args()


def main():
    global args

    # create submit dir
    os.makedirs('./submit', exist_ok=True)

    # create datasets
    test_dataset = dsets.CARVANA(root=args.dir,
                                 train=False,
                                 transform=unet_transforms.Compose([
                                     unet_transforms.Scale((256, 256)),
                                     unet_transforms.ToNumpy(),
                                     unet_transforms.ToTensor()
                                 ])
                                 )
    # define the dataloader with the previous dataset
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              drop_last=False,
                                              shuffle=False,
                                              num_workers=args.workers)

    to_unet = transform = unet_transforms.Compose([
        unet_transforms.Scale((256, 256)),
        unet_transforms.ToNumpy(),
        unet_transforms.ToTensor(),
    ])

    # change model to cuda and set to evaluation mode
    model = models.UNet1024().cuda()

    # create transform to resize the image output
    orig_size = (1920, 1280)
    to_orig_size = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Scale(orig_size, interpolation=Image.BILINEAR),
        transforms.ToTensor()
    ])

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # set model to evaluation
    model.eval()

    # create the file path
    f = open(args.submit_dir + args.arch + '.csv', 'w')
    f.write('img,rle_mask\n')  # python will convert \n to os.linesep

    # run test loop with a progress bar
    for i, (images, _) in tqdm(enumerate(test_loader), total=len(test_loader)):
        # Convert torch tensor to cuda Variable
        images = Variable(images.cuda(), volatile=True)

        # compute sigmoid(output) and move to cpu
        outputs = model(images)
        outputs = F.sigmoid(outputs).data.cpu()

        for j in range(outputs.size(0)):
            # print('\n' + str(test_dataset.data_names[j]))

            # change output to a 3D tensor (1, height, width), rescale to the original size and mask it
            new_output = (to_orig_size(outputs[j].view(1, outputs[j].size(0), outputs[j].size(1))) > 0.5)
            # data_utils.im_show([to_orig_size(torch.squeeze(images.data[j].cpu())), new_output])

            # run the rle with the mask and append it with the proper format
            # print(test_dataset.data_names[j] + ',' + data_utils.mask_to_RLEstring(new_output.numpy()))
            f.write(test_dataset.data_names[j] + ',' + data_utils.mask_to_RLEstring(new_output.numpy()) + '\n')

    # close the open file
    f.close()


if __name__ == "__main__":
    main()
