import argparse
import os

import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import datasets as dsets
import models
from datasets import transforms as unet_transforms
from datasets import utils as data_utils

# from tf_logger import Logger

# define the parser arguments
parser = argparse.ArgumentParser(description='Car-segmentation kaggle competition')

parser.add_argument('--dir', default="./data", type=str, metavar='D',
                    help='path to the dataset root folder (default: ./data/')
parser.add_argument('--submit_dir', default='./submit/', type=str, metavar='L',
                    help='path to the folder to store the submit file')

parser.add_argument('--arch', '-a', metavar='ARCH', default='UNET_1024_ShiftScaleRotate',
                    help='name of the architecture used, also used to store the submit file')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')

parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--resume', default='./checkpoints/UNET_1024_ShiftScaleRotatemodel_best.pth.tar', type=str,
                    metavar='PATH', help='path to latest checkpoint')

global args
args = parser.parse_args()


def main():
    global args

    # set csv block size
    CSV_BLOCK_SIZE = 16000

    # create submit dir
    os.makedirs('./submit', exist_ok=True)

    # create datasets
    test_dataset = dsets.CARVANA(root=args.dir,
                                 subset="test",
                                 transform=unet_transforms.Compose([
                                     unet_transforms.Scale((1024, 1024)),
                                     unet_transforms.ToNumpy(),
                                     unet_transforms.ToTensor(),
                                 ])
                                 )
    # define the dataloader with the previous dataset
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              drop_last=False,
                                              pin_memory=True,
                                              num_workers=4)

    # change model to cuda and set to evaluation mode
    model = models.UNet1024().cuda()
    model.eval()

    # create transform for the image output
    to_orig_size = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Scale((1920, 1280)),
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

    # crete the array to store the output
    rle = []

    # set a progress bar
    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    for i, (images, _) in pbar:
        # Convert torch tensor to cuda Variable
        images = Variable(images.cuda())

        # compute output
        outputs = model(images)
        outputs = to_orig_size(outputs.data.cpu())

        mask = (outputs > 0.5).numpy()
        rle.append(data_utils.rle_encode(mask))

    df = pd.DataFrame({'img': test_dataset.data_names, 'rle_mask': rle})
    df.to_csv(args.submit_dir + args.arch + 'csv.gz', index=False,
              compression='gzip')


if __name__ == "__main__":
    main()
