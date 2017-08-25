import gc
import argparse
import datasets as dsets
import models
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import datasets.utils as data_utils
import models.utils as model_utils
from tf_logger import Logger
from tqdm import tqdm

# define the parser arguments
parser = argparse.ArgumentParser(description='Car-segmentation kaggle competition')

parser.add_argument('--dir', default="./data/", type=str, metavar='D',
                    help='path to the dataset root folder (default: ./data/')
parser.add_argument('--logs_dir', default='./logs/', type=str, metavar='L',
                    help='path to the logs folder (default: ./logs/')

parser.add_argument('--arch', '-a', metavar='ARCH', default='UNET_256')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')

parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--nesterov', default=True, type=bool, metavar='N', help='use nesterov momentum (default:True)')

parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--test', dest='test', action='store_true', help='evaluate model on test set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')

global args, best_loss
args = parser.parse_args()
best_loss = 0

# create datasets
train_dataset = dsets.CARVANA(root=args.dir,
                              subset="train",
                              transform=transforms.Compose([
                                  transforms.Scale(256),
                                  transforms.RandomCrop(256),
                                  transforms.ToTensor()])
                              )

val_dataset = dsets.CARVANA(root=args.dir,
                            subset="val",
                            transform=transforms.Compose([
                                transforms.Scale(256),
                                transforms.RandomCrop(256),
                                transforms.ToTensor()])
                            )

# define the dataloader with the previous dataset
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=1)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=1)

# create model, define the loss function and the optimizer.
# Move everything to cuda
model = models.small_UNET_256().cuda()
criterion = models.BCELoss2d().cuda()
optimizer = optim.SGD(model.parameters(),
                      weight_decay=args.weight_decay,
                      lr=args.lr,
                      momentum=args.momentum,
                      nesterov=args.nesterov)

# Set the logger
logger = Logger(args.logs_dir)


# define the training function
def train(train_loader, model, criterion, epoch):
    model.train()
    losses = data_utils.AverageMeter()

    # set a progress bar
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in pbar:
        # Convert torch tensor to Variable
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

        # compute output
        optimizer.zero_grad()
        outputs = model(images)

        # measure loss
        loss = criterion(outputs, labels)
        losses.update(loss.data[0], images.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # logging
        logger.scalar_summary('(train)loss_val', losses.val, i + 1)
        logger.scalar_summary('(train)loss_avg', losses.avg, i + 1)

        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary('(train)' + tag, data_utils.to_np(value), i + 1)
            logger.histo_summary('(train)' + tag + '/grad', data_utils.to_np(value.grad), i + 1)
        #
        # logger.image_summary('[TRAIN] model outputs', data_utils.to_np(outputs), epoch + 1)

        # update progress bar status
        pbar.set_description('[TRAIN] - EPOCH %d/ %d - BATCH LOSS: %.4f/ %.4f(avg) '
                             % (epoch + 1, args.epochs, losses.val, losses.avg))


# define the validation function
def validate(val_loader, model, criterion, epoch):
    model.eval()
    losses = data_utils.AverageMeter()

    # set a progress bar
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for i, (images, labels) in pbar:
        # Convert torch tensor to Variable
        images = Variable(images.cuda(), volatile=True)
        labels = Variable(labels.cuda(), volatile=True)

        # compute output
        outputs = model(images)

        # measure loss and log it
        loss = criterion(outputs, labels)
        losses.update(loss.data[0], images.size(0))

        # logging
        logger.scalar_summary('data/(val)loss_val', losses.val, i + 1)
        logger.scalar_summary('data/(val)loss_avg', losses.avg, i + 1)

        # for tag, value in model.named_parameters():
        #     tag = tag.replace('.', '/')
        #     logger.histo_summary('[VAL] ' + tag, data_utils.to_np(value), epoch + 1)
        #     logger.histo_summary('[VAL] ' + tag + '/grad', data_utils.to_np(value.grad), epoch + 1)
        #
        logger.image_summary('(val)model_output', data_utils.to_np(outputs), epoch + 1)

        # update progress bar status
        pbar.set_description('[VAL] - EPOCH %d/ %d - BATCH LOSS: %.4f/ %.4f(avg) '
                             % (epoch + 1, args.epochs, losses.val, losses.avg))

    return losses.avg


# run the training loop
for epoch in range(args.start_epoch, args.epochs):
    # adjust lr according to the current epoch
    model_utils.adjust_learning_rate(optimizer, epoch, args.lr)

    # train for one epoch
    train(train_loader, model, criterion, epoch)

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
