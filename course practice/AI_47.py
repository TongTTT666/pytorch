import argparse

import shutil
import time

# package from pytorch
import torch
import torchvision.models as models
from torch import nn
import torch.optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

# Add  debug command
# Training settings: we can set the parameters using terminal instead of IDE
# This is a very simple way to debug my codes. It is convenient to edit the training parameters
# Create a argparse object
parser = argparse.ArgumentParser(
    description='This is the big assignment of AI class(Author AI-47)')

# set parameters
parser.add_argument(
    '--batch-size',
    type=int,
    default=64,
    metavar='N',
    help='input batch size for training (default: 64)')
parser.add_argument(
    '--test-batch-size',
    type=int,
    default=100,
    metavar='N',
    help='input batch size for testing (default: 100)')
parser.add_argument(
    '--epochs',
    type=int,
    default=10,
    metavar='N',
    help='number of epochs to train (default: 10)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    type=float,
    default=0.01,
    metavar='LR',
    help='learning rate (default: 0.01)')
parser.add_argument(
    '--momentum',
    type=float,
    default=0.5,
    metavar='M',
    help='SGD momentum (default: 0.5)')
parser.add_argument(
    '--disable-cuda',
    action='store_true',
    default=False,  # If "action" is existing, "metavar" must not appear
    help='disables CUDA training')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='how many batches to wait before logging training status (default: 10)'
)
parser.add_argument(
    '--pretrained',
    action='store_true',
    default=False,
    help='use pre-trained model')
parser.add_argument(
    '--is-train',
    action='store_false',
    default=True,
    help='determine whether to train')
parser.add_argument(
    '--disable-adjust-learning-rate',
    '--dalr',
    action='store_false',
    default=True,
    help='Choose whether to start up  the automatic adjust of learning rate')
parser.add_argument(
    '--model-name',
    '--mn',
    metavar='NAME',
    default='resnet18',
    choices=model_names,
    help='model architecture: ' + ' | '.join(model_names) +
    ' (default: resnet18)')
parser.add_argument(
    '--disable-checkpoint',
    action='store_false',
    default=True,
    help='disable the checkpoint to save the current best model')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=0,
    type=float,
    metavar='W',
    help='weight decay (default: 0)')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '--gpu-device',
    nargs="+",
    type=int,
    metavar='LIST',
    help='choose gpu device(you can choose none, one or more gpus)')
parser.add_argument(
    '--evaluate',
    '-e',
    action='store_true',
    default=False,
    help='evaluate model on validation set')

args = parser.parse_args()
# Obtain a cuda variable to determine whether to use cuda accelerate
args.cuda = not args.disable_cuda and torch.cuda.is_available()


def main():
    best_prec1 = 0
    # record the history precision of test dataset
    prec1_arr = []
    # record the history precision and loss of training dataset(top1)
    his_prec = []
    his_loss = []

    # create model (whether to use pre-trained model)
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.model_name))
        model = models.__dict__[args.model_name](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.model_name))
        model = models.__dict__[args.model_name]()

    # define whether to use cuda accelerate
    if args.cuda:
        model = torch.nn.DataParallel(model)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            prec1_arr = checkpoint['prec1_arr']
            his_loss = checkpoint['his_loss']
            his_prec = checkpoint['his_prec']
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.cuda:
        model.cuda()
        cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    if args.cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    # optionally resume from a optimizer
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading optimizer '{}'".format(args.resume))
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded optimizer '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no optimizer found at '{}'".format(args.resume))

    # Data loading code. According to whether to use pretrained model, there are two types of transforms
    if args.pretrained:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform1 = transforms.Compose([
            transforms.RandomResizedCrop(224),  # must resize to 224*224
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform2 = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # transform1 = transforms.Compose(
        #     [transforms.ToTensor(),
        #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # transform2 = transforms.Compose(
        #     [transforms.ToTensor(),
        #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform1 = transforms.Compose([
            transforms.RandomResizedCrop(224),  # must be resize to 224*224
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform2 = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform1)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform2)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')

    if args.evaluate:
        validate(test_loader, model, criterion)
        return  # end the codes

    # output some parameters
    print('Parameters in the training process\n'
          'Batch size: {0}\n'
          'Test Batch size: {1}\n'
          'Maximum epoch: {2}\n'
          'Log interval: {3}\n'.format(args.batch_size, args.test_batch_size,
                                       args.epochs, args.log_interval))

    for epoch in range(args.start_epoch, args.epochs):
        if args.disable_adjust_learning_rate:
            adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, his_prec,
              his_loss)

        # evaluate on validation set
        prec1 = validate(test_loader, model, criterion)
        # save the prec1 in testset
        prec1_arr.append(prec1)

        if args.disable_checkpoint:
            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.model_name,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
                'prec1_arr': prec1_arr,
                'his_prec': his_prec,
                'his_loss': his_loss
            }, is_best)

        # print(his_prec)
        # print(his_loss)
        # print(prec1_arr)

    # Save the final result as a file
    print('Save the final result as a file')
    prec1_arr = np.array(prec1_arr)
    his_prec = np.array(his_prec)
    his_loss = np.array(his_loss)
    np.save("prec1_arr.npy", prec1_arr)
    np.save("his_prec.npy", his_prec)
    np.save("his_loss.npy", his_loss)


def train(train_loader, model, criterion, optimizer, epoch, his_prec,
          his_loss):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch the model to the train mode
    model.train()

    end = time.time()
    # input the dataset
    for index, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_var, target_var = torch.autograd.Variable(
            input.cuda()), torch.autograd.Variable(target.cuda())

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss (top1 and top5)
        prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # bp process
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if index % args.log_interval == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch,
                      index,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1,
                      top5=top5))

        if index == len(
                train_loader) - 1:  #save history information in the last batch
            print('Save history information')
            his_prec.append(top1.avg)
            his_loss.append(losses.avg)


def validate(test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch the model to the  mode
    model.eval()

    end = time.time()
    # input the dataset
    for index, (input, target) in enumerate(test_loader):

        input_var, target_var = torch.autograd.Variable(
            input.cuda()), torch.autograd.Variable(target.cuda())

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss (top1 and top5)
        prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if index % args.log_interval == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      index,
                      len(test_loader),
                      batch_time=batch_time,
                      loss=losses,
                      top1=top1,
                      top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(
        top1=top1, top5=top5))

    return top1.avg


def adjust_learning_rate(optimizer, epoch, decay_ratio=0.1, epoch_num=30):
    """Sets the learning rate to the initial LR decayed by decay_rate every epoch_num epochs"""
    lr = args.lr * (decay_ratio**(epoch // epoch_num))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        print('This model is the best model, save it to another file')
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
