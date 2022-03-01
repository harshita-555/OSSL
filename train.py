import argparse
import os
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from torch.utils.tensorboard import SummaryWriter

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils import rotate_batch
import densenet as dn

parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--log_dir', default="./logs/")
parser.add_argument('--epochs', default=300, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--name', default='DenseNet-40-12-ss', type=str,
                    help='name of experiment')
parser.add_argument('--layers', default=40, type=int,
                    help='total number of layers (default: 40)')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--reduce', default=1.0, type=float,
                    help='compression rate in transition stage (default: 1.0)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
parser.add_argument('--resume', default='./runs/DenseNet-40-12-ss/checkpoint.pth.tar', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--rotation_type', default='expand')
parser.add_argument('--rot', action='store_true', default=False)
parser.add_argument('--two_classifiers', action='store_true', default=False)
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    writer = SummaryWriter(log_dir=args.log_dir)
    kwargs = {'num_workers': 4, 'pin_memory': True}

    # Data loading code
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
                         transform=transform_train),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transform_test),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    from cifar_new import CIFAR_New
    print('Test on CIFAR-10.1')
    teset = CIFAR_New(root='./data/' + 'CIFAR-10.1/', transform=transform_test)
    test_loader = torch.utils.data.DataLoader(teset, batch_size=64,
                                             shuffle=False, num_workers=4)

    # create model
    model = dn.DenseNet3(args.layers, 10, args.growth, reduction=args.reduce,
                         bottleneck=args.bottleneck, dropRate=args.droprate)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
    else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)

    complement_optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):
         adjust_learning_rate(optimizer, epoch)
         adjust_learning_rate(complement_optimizer, epoch)

         # train for one epoch
         prec1, loss1, prec2, loss2 = train(train_loader, model, criterion, optimizer, complement_optimizer, epoch)

         writer.add_scalar('Loss/train_cls1', loss1, epoch)
         writer.add_scalar('Loss/train_cls2', loss2, epoch)
         writer.add_scalar('Accuracy/train_cls1', prec1, epoch)
         writer.add_scalar('Accuracy/train_cls2', prec2, epoch)

         # evaluate on test set for semantic classification
         prec1, loss1, prec2, loss2 = validate(test_loader, model, criterion)

         writer.add_scalar('Loss/test_cls1', loss1, epoch)
         writer.add_scalar('Loss/test_cls2', loss2, epoch)
         writer.add_scalar('Accuracy/test_cls1', prec1, epoch)
         writer.add_scalar('Accuracy/test_cls2', prec2, epoch)

         if epoch >= 49 and (epoch + 1) % 10 == 0:
              # remember best prec@1 and save checkpoint
              if prec1 >= prec2:
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
              if prec2 > prec1:
                is_best = prec2 > best_prec1
                best_prec1 = max(prec2, best_prec1)
              save_checkpoint({
                  'epoch': epoch + 1,
                  'state_dict': model.state_dict(),
                  'best_prec1': best_prec1,
              }, is_best)
    print('Best accuracy: ', best_prec1)


def train(train_loader, model, criterion, optimizer, complement_optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    prec1 = AverageMeter()
    prec2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda()
        input = input.cuda()

        # compute output
        cls1, _, _ = model(input)
        loss = criterion(cls1, target) 

        # self-supervised
        if args. rot :
            inputs_rot, labels_rot = rotate_batch(input, args.rotation_type)
            inputs_rot, labels_rot = inputs_rot.cuda(), labels_rot.cuda()
            _, _, outputs_rot = model(inputs_rot)
            loss  += criterion(outputs_rot, labels_rot)


        # measure accuracy and record loss
        prec = accuracy(cls1, target, topk=(1,))[0]
        losses1.update(loss.item(), input.size(0))
        prec1.update(prec.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.two_classifiers :
            cls1, cls2, _ = model(input)
            loss1 = criterion(cls1.detach(), target)
            loss2  = criterion(cls2, target)

            lossx = loss2 - loss1 + 1e-7

            prec = accuracy(cls2, target, topk=(1,))[0]
            losses2.update(loss2.item(), input.size(0))
            prec2.update(prec.item(), input.size(0))

            complement_optimizer.zero_grad()
            lossx.backward()
            complement_optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss_Cls {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses1, top1=prec1))

    return prec1.avg, losses1.avg, prec2.avg, losses2.avg


def validate(val_loader, model, criterion):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    top1_1 = AverageMeter()
    top1_2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()

        # compute output
        with torch.no_grad():
            cls1, cls2, _ = model(input)
            loss1 = criterion(cls1, target)
            loss2 = criterion(cls2, target)
            # measure accuracy and record loss
            prec1 = accuracy(cls1, target, topk=(1,))[0]
            prec2 = accuracy(cls2, target, topk=(1,))[0]
            losses1.update(loss1.item(), input.size(0))
            losses2.update(loss2.item(), input.size(0))
            top1_1.update(prec1.item(), input.size(0))
            top1_2.update(prec2.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses1,
                top1=top1_1))

    print(' * Prec@1  {top1.avg:.3f}, {top2.avg:.3f}'.format(top1=top1_1,top2=top1_2))

    return top1_1.avg, losses1.avg, top1_2.avg, losses2.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/" % (args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (args.name) + 'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 130 and 200 epochs"""
    lr = args.lr * (0.1 ** (epoch // 130)) * (0.1 ** (epoch // 200))
    # log to TensorBoard
    # if args.tensorboard:
    #     log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()