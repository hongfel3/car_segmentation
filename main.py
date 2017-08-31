import argparse

import torch
import torch.optim as optim
from tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
# from tf_logger import Logger
from tqdm import tqdm

import datasets as dsets
import datasets.utils as data_utils
import models
import models.utils as model_utils
from datasets import transforms as unet_transforms

# define the parser arguments
parser = argparse.ArgumentParser(description='Car-segmentation kaggle competition')

parser.add_argument('--dir', default="./data/", type=str, metavar='D',
                    help='path to the dataset root folder (default: ./data/')
parser.add_argument('--logs_dir', default=None, type=str, metavar='L',
                    help='path to the logs folder (default: ./logs/')

parser.add_argument('--arch', '-a', metavar='ARCH', default='UNET_256')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')

parser.add_argument('--epochs', default=45, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', default=5e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--nesterov', default=False, type=bool, metavar='N', help='use nesterov momentum (default:True)')
parser.add_argument('--batch_acum', default=4, type=bool, metavar='B',
                    help='number of iterations the batch is accumulated')

parser.add_argument('--model_log', default=500, type=int, metavar='I', help='number of iterations between logs')

parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--test', dest='test', action='store_true', help='evaluate model on test set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')

global args, best_loss, logger, graph_logged
args = parser.parse_args()
best_loss = 0
logger = SummaryWriter(log_dir=args.logs_dir, comment=args.arch)
graph_logged = False


def main():
    global args, best_loss

    # create datasets
    train_dataset = dsets.CARVANA(root=args.dir,
                                  subset="train",
                                  transform=unet_transforms.Compose([
                                      unet_transforms.Scale((1024,1024)),
                                      unet_transforms.ToNumpy(),
                                      unet_transforms.RandomShiftScaleRotate(),
                                      unet_transforms.ToTensor(),
                                      # unet_transforms.RandomCrop(1024),
                                  ])
                                  )

    val_dataset = dsets.CARVANA(root=args.dir,
                                subset="val",
                                transform=transforms.Compose([
                                    unet_transforms.Scale((1024,1024)),
                                    # unet_transforms.RandomCrop(1024),
                                    unet_transforms.ToTensor()])
                                )

    # define the dataloader with the previous dataset
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=4)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=4)

    # create model, define the loss function and the optimizer.
    # Move everything to cuda
    model = models.UNet1024().cuda()
    criterion = {'loss': models.weightedBCEplusDice().cuda(),
                 'acc': models.diceLoss().cuda()}
    optimizer = optim.SGD(model.parameters(),
                          weight_decay=args.weight_decay,
                          lr=args.lr,
                          momentum=args.momentum,
                          nesterov=args.nesterov)

    # run the training loop
    for epoch in range(args.start_epoch, args.epochs):
        # adjust lr according to the current epoch
        model_utils.adjust_learning_rate(optimizer, epoch, args.lr)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate the model
        curr_loss = validate(val_loader, model, criterion, epoch)

        # store best loss and save a model checkpoint
        is_best = curr_loss < best_loss
        best_loss = max(curr_loss, best_loss)
        data_utils.save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best)

    logger.close()


# define the training function
def train(train_loader, model, criterion, optimizer, epoch):
    global args, logger, graph_logged

    model.train()
    losses = data_utils.AverageMeter()
    accuracy = data_utils.AverageMeter()

    batch_acum = 0
    # set a progress bar
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in pbar:
        # Convert torch tensor to Variable
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        # print(labels.cpu().data)
        # print (np.histogram(labels.cpu().data.numpy()))

        # compute output
        optimizer.zero_grad()
        outputs = model(images)

        # measure loss
        loss = criterion['loss'](outputs, labels)
        losses.update(loss.data[0], images.size(0))

        # get accuracy
        mask = (outputs.data > 0.5).float()
        acc = criterion['acc'](mask, labels.data)
        accuracy.update(acc, images.size(0))

        # compute gradient and do SGD step
        # if args.batch_acum != 0 and batch_acum % args.batch_acum == 0:
        loss.backward()
        optimizer.step()
        batch_acum += 1

        # logging
        if not graph_logged:
            logger.add_graph(model, outputs)
            graph_logged = True

        logger.add_scalar('(train)loss_val', losses.val, i + 1)
        logger.add_scalar('(train)loss_avg', losses.avg, i + 1)

        if i % args.model_log == 0:
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.add_histogram('model/(train)' + tag, data_utils.to_np(value), i + 1)
                logger.add_histogram('model/(train)' + tag + '/grad', data_utils.to_np(value.grad), i + 1)

        logger.add_image('model/(train)output', make_grid(mask.float(), scale_each=True), i + 1)

        # update progress bar status
        pbar.set_description('EPOCH %d/ %d - LOSS %.4f/ %.4f(avg) - ACC %.4f/ %.4f(avg)'
                             % (epoch + 1, args.epochs, losses.val, losses.avg, accuracy.val,
                                accuracy.avg))  # define the validation function


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

        mask = (outputs.data > 0.5).float()
        acc = criterion['acc'](mask, labels.data)
        accuracy.update(acc, images.size(0))

        # logging
        logger.add_scalar('data/(val)loss_val', losses.val, i + 1)
        logger.add_scalar('data/(val)loss_avg', losses.avg, i + 1)

        logger.add_image('model/(val)output', make_grid(mask), i + 1)

        # update progress bar status
        pbar.set_description('EPOCH %d/ %d - LOSS %.4f/ %.4f(avg) - ACC %.4f/ %.4f(avg)'
                             % (epoch + 1, args.epochs, losses.val, losses.avg, accuracy.val,
                                accuracy.avg))

    return losses.avg


if __name__ == "__main__":
    main()
