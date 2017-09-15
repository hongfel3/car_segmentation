import argparse
import os

import torch
import torch.optim as optim
from tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import numpy as np
torch.backends.cudnn.benchmark = True  # enables cudnn's auto-tuner

from tqdm import tqdm

import datasets as dsets
import datasets.utils as data_utils
import models
import models.utils as model_utils
from datasets import transforms_unet as unet_transforms
from datasets import samplers as data_samplers

# define the parser arguments
parser = argparse.ArgumentParser(description='Car-segmentation kaggle competition')

parser.add_argument('--dir', default="./data/", type=str, metavar='D',
                    help='path to the dataset root folder (default: ./data/')
parser.add_argument('--logs_dir', default=None, type=str, metavar='L',
                    help='path to the logs folder (default: ./logs/')

parser.add_argument('--arch', '-a', metavar='ARCH', default='UNET_256_ShiftScaleRotate_noweightedLoss')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')

parser.add_argument('--epochs', default=45, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--adjust-epoch', default=10, type=int, metavar='E',
                    help='Number of epochs to run to adjust the learning rate')
parser.add_argument('--perc-val', default=0.1, type=int, metavar='V',
                    help='percentage of the train set used for validation')

parser.add_argument('-b', '--batch-size', default=20, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', default=5e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--nesterov', dest='nesterov', action='store_true', help='use nesterov momentum (default:True)')
parser.add_argument('--batch_acum', default=0, type=bool, metavar='B',
                    help='number of iterations the batch is accumulated')

parser.add_argument('--model_log', default=500, type=int, metavar='I', help='number of iterations between logs')

parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')

global args, best_loss, logger, graph_logged
args = parser.parse_args()
best_loss = 0
logger = SummaryWriter(log_dir=args.logs_dir, comment=args.arch)
graph_logged = False


def main():
    global args, best_loss

    # create necessary folders
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./runs', exist_ok=True)

    # create datasets
    train_dataset = dsets.CARVANA(root=args.dir,
                                  train=True,
                                  transform=unet_transforms.Compose([
                                      unet_transforms.Scale((256, 256)),
                                      unet_transforms.ToNumpy(),
                                      unet_transforms.RandomShiftScaleRotate(),
                                      unet_transforms.ToTensor(),
                                  ])
                                  )

    val_dataset = dsets.CARVANA(root=args.dir,
                                train=True,
                                transform=unet_transforms.Compose([
                                    unet_transforms.Scale((256, 256)),
                                    unet_transforms.ToNumpy(),
                                    unet_transforms.ToTensor()])
                                )

    num_val = int(len(train_dataset) * args.perc_val)
    num_train = len(train_dataset) - num_val

    # define the dataloader with the previous dataset
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               # sampler=data_samplers.ChunkSampler(num_train, start=0),
                                               num_workers=4)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             sampler=data_samplers.ChunkSampler(num_val, start=num_train),
                                             num_workers=4)

    # create model, define the loss function and the optimizer.
    # Move everything to cuda


    model = models.UNet1024().cuda()

    criterion = {'loss': models.BCEplusDice().cuda(),
                 'acc': models.diceAcc().cuda()}
    optimizer = optim.SGD(model.parameters(),
                          weight_decay=args.weight_decay,
                          lr=args.lr,
                          momentum=args.momentum,
                          nesterov=args.nesterov)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(val_loader, model, criterion, checkpoint['epoch'])
        return

    # run the training loop
    for epoch in range(args.start_epoch, args.epochs):
        # adjust lr according to the current epoch
        model_utils.adjust_learning_rate(optimizer, epoch, args.lr, args.adjust_epoch)

        # train for one epoch
        curr_loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate the model
        # curr_loss = validate(val_loader, model, criterion, epoch)

        # store best loss and save a model checkpoint
        is_best = curr_loss < best_loss
        best_loss = max(curr_loss, best_loss)
        data_utils.save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, folder='./checkpoints/', filename=args.arch)

    logger.close()


def train(train_loader, model, criterion, optimizer, epoch):
    global args, logger, graph_logged

    # set model to train
    model.train()

    # set initial vars
    losses = data_utils.AverageMeter()
    accuracy = data_utils.AverageMeter()
    batch_acum = 0

    # set a progress bar
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in pbar:

        # Convert torch tensor to cuda Variable
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

        # print(labels.cpu().data)
        # print (np.histogram(labels.cpu().data.numpy()))

        # compute output
        outputs = model(images)

        # measure loss
        loss = criterion['loss'](outputs, labels)
        losses.update(loss.data[0], images.size(0))

        # compute gradient
        loss.backward()

        # take a step if we reach the batch accumulations
        if args.batch_acum == 0 or batch_acum % args.batch_acum == 0:
            optimizer.step()
            optimizer.zero_grad()

        batch_acum += 1

        # get accuracy
        mask = (outputs.data > 0.5).float()
        mask_target = (labels.data > 0.5).float()
        acc = criterion['acc'](mask, mask_target)
        accuracy.update(acc, images.size(0))

        # logging
        if not graph_logged:
            logger.add_graph(model, outputs)
            graph_logged = True

        logger.add_scalar('(train)loss_val', losses.val, (epoch * len(train_loader)) + i + 1)
        logger.add_scalar('(train)loss_avg', losses.avg, (epoch * len(train_loader)) + i + 1)

        if i % args.model_log == 0:
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.add_histogram('model/(train)' + tag, data_utils.to_np(value), i + 1)
                logger.add_histogram('model/(train)' + tag + '/grad', data_utils.to_np(value.grad), i + 1)

        idx = np.random.randint(0, len(labels))
        logger.add_image('model/(train)output', make_grid(mask[idx], scale_each=True), i + 1)
        logger.add_image('model/(train)target', make_grid(mask_target[idx], scale_each=True), i + 1)

        # update progress bar status
        pbar.set_description('EPOCH %d/ %d - LOSS %.4f/ %.4f(avg) - ACC %.4f/ %.4f(avg)'
                             % (epoch + 1, args.epochs, losses.val, losses.avg, accuracy.val,
                                accuracy.avg))  # define the validation function
    return losses.avg


def validate(val_loader, model, criterion, epoch):
    global args, logger
    model.eval()
    losses = data_utils.AverageMeter()
    accuracy = data_utils.AverageMeter()

    # set a progress bar
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for i, (images, labels) in pbar:
        # Convert torch tensor to Variable
        images = Variable(images.cuda(), volatile=True)
        labels = Variable(labels.cuda(), volatile=True)

        # compute output
        outputs = model(images)

        # measure loss, accuracy and log it
        loss = criterion['loss'](outputs, labels)
        losses.update(loss.data[0], images.size(0))

        # get accuracy
        mask = (outputs.data > 0.5).float()
        mask_target = (labels.data > 0.5).float()
        acc = criterion['acc'](mask, mask_target)
        accuracy.update(acc, images.size(0))

        # logging
        logger.add_scalar('data/(val)loss_val', losses.val, i + 1)
        logger.add_scalar('data/(val)loss_avg', losses.avg, i + 1)

        logger.add_image('model/(val)output', make_grid(mask), i + 1)
        logger.add_image('model/(val)target', make_grid(mask_target, scale_each=True), i + 1)

        # update progress bar status
        pbar.set_description('EPOCH %d/ %d - LOSS %.4f/ %.4f(avg) - ACC %.4f/ %.4f(avg)'
                             % (epoch + 1, args.epochs, losses.val, losses.avg, accuracy.val,
                                accuracy.avg))

    return losses.avg


if __name__ == "__main__":
    main()
