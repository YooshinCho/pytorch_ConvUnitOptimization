from __future__ import print_function

import argparse
import os
import shutil
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models as models
import utils.tools as tools
import utils.function as func
from utils.logger import Logger

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--datadir', default='./data', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                                        help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                                        help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                                        help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                                        help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                                        help='test batchsize')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                                        metavar='Dropout', help='Dropout ratio')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                                        metavar='LR', help='initial learning rate')
parser.add_argument('--nstrv', action='store_true',
                                        help='nesterov momentum')
parser.add_argument('--cos', action='store_true',
                                        help='cosine anealing')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                                                help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                                        help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                                        metavar='W', help='weight decay (default: 1e-4)')



parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                                        help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                                        help='path to latest checkpoint (default: none)')
parser.add_argument('--transfer', action='store_true', help='transfer or not')



parser.add_argument('--arch', '-a', metavar='ARCH')
parser.add_argument('--depth', type=int, default=20, help='Model depth.')
parser.add_argument('--widen-factor', type=int, default=10, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--bias', action='store_true', help='Conv bias')
parser.add_argument('--df', action='store_true', help='IterNorm with DF')
parser.add_argument('--norm-type', type=str, default = 'batchnorm', help='normalization type')
parser.add_argument('--unit-type' , type=str, default = 'org', help='convolutional unit type')
parser.add_argument('--norm-cfg' , metavar='DICT', default={}, type=tools.str2dict, help='argument of normalization layer')
parser.add_argument('--expansion', type=int, default=6, help='expansion rate of shiftresnet')
parser.add_argument('--mult', type=float, default=1, help='multiplier of depth, width of shiftnet-A')


parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                                        help='evaluate model on validation set')
parser.add_argument('--gpu-id', default='0', type=str,
                                        help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--freq', type=int, default=90, help='print frequency')

args = parser.parse_args()
print(args)
        
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    
    if not os.path.isdir(args.checkpoint):
        tools.mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
            
    if args.dataset.startswith('cifar'):
        if args.dataset == 'cifar10':
            dataloader = datasets.CIFAR10
            num_classes = 10
        elif args.dataset == 'cifar100':
            dataloader = datasets.CIFAR100
            num_classes = 100
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        trainset = dataloader(root=args.datadir, train=True, download=True, transform=transform_train)
        
        trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, pin_memory =True)

        testset = dataloader(root=args.datadir, train=False, download=False, transform=transform_test)
        testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory =True)
    
    
    
    elif args.dataset == 'imagenet':
        num_classes = 1000
        traindir = os.path.join(args.datadir, 'train')
        valdir = os.path.join(args.datadir, 'val')
        
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,
                    ])   
        transform_test = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                    ])
                    
                    
        train_dataset = datasets.ImageFolder(traindir, transform_train)
        test_dataset = datasets.ImageFolder(valdir, transform_test) 
        

        trainloader = data.DataLoader(
                train_dataset, batch_size=args.train_batch, shuffle=True,
                num_workers=args.workers, pin_memory=True)


        testloader = data.DataLoader(
                test_dataset, batch_size=args.test_batch, shuffle=False,
                num_workers=args.workers, pin_memory=True)                                      

    elif args.dataset == 'dogs':
        num_classes = 1000
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                             std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
                            transforms.RandomResizedCrop(448),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,
                    ])   
        transform_test = transforms.Compose([
                            transforms.Resize(512),
                            transforms.CenterCrop(448),
                            transforms.ToTensor(),
                            normalize,
                    ])
                    
                    
        train_dataset = tools.dogs(args.datadir, train=True,transform=transform_train)
        test_dataset = tools.dogs(args.datadir,train=False,transform=transform_test) 

        trainloader = data.DataLoader(
                train_dataset, batch_size=args.train_batch, shuffle=True,
                num_workers=args.workers, pin_memory=True)


        testloader = data.DataLoader(
                test_dataset, batch_size=args.test_batch, shuffle=False,
                num_workers=args.workers, pin_memory=True)                                      
    elif args.dataset == 'cub':
        num_classes = 1000
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                             std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
                            transforms.RandomResizedCrop(448),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,
                    ])   
        transform_test = transforms.Compose([
                            transforms.Resize(512),
                            transforms.CenterCrop(448),
                            transforms.ToTensor(),
                            normalize,
                    ])
                    
                    
        train_dataset = tools.Cub2011(args.datadir, train=True,transform=transform_train)
        test_dataset = tools.Cub2011(args.datadir,train=False,transform=transform_test) 
        

        trainloader = data.DataLoader(
                train_dataset, batch_size=args.train_batch, shuffle=True,
                num_workers=args.workers, pin_memory=True)


        testloader = data.DataLoader(
                test_dataset, batch_size=args.test_batch, shuffle=False,
                num_workers=args.workers, pin_memory=True)                                      

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                                    args = args,
                                    num_classes=num_classes,
                                    depth=args.depth,
                                    widen_factor=args.widen_factor,
                                    dropRate=args.drop,
                            )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                                    num_classes=num_classes,
                                    depth=args.depth,
                                    args = args
                            )
    elif args.arch.startswith('shiftneta'):
        model = models.__dict__[args.arch](
                                    args = args,
                                    num_classes=num_classes
                            )
    else:
        model = models.__dict__[args.arch](args=args,num_classes=num_classes)

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    
    # Loss function, Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov = args.nstrv)
    if not args.cos:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = args.schedule, gamma = args.gamma)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs*len(trainloader))


    # Resume
    title = args.dataset + '/' + args.arch
    names = ['Epoch','Train_Loss','Train_Acc', 'Test_loss','Test_Acc']
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'], strict = not args.transfer)
        
        # Load optimizer, scheduler
        if not args.transfer:
            best_acc = checkpoint['best_acc']
            start_epoch = checkpoint['epoch']
            print('Best accuracy: {}, Start epoch: {}'.format(best_acc, start_epoch))
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            if os.path.isfile(os.path.join(args.checkpoint, 'log.txt')):
                logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume = True)
            
        # Transfer Learning
        if args.transfer:
            if args.dataset == 'cub':
                model.module.fc = nn.Linear(model.module.fc.in_features,200).cuda()
            if args.dataset == 'dogs':
                model.module.fc = nn.Linear(model.module.fc.in_features,120).cuda()

            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov = args.nstrv)
            if not args.cos:
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = args.schedule, gamma = args.gamma)
            else:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs*len(trainloader))
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
            logger.set_names(names)
        
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(names)
            
    print(model)
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))


    if args.evaluate:
        print('\nEvaluation only')
        func.evaluate(testloader, model, criterion, start_epoch,use_cuda, args)
        return
            
    
    # Train 
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        
        tr_loss,tr_top1,tr_top5 = func.train(trainloader, model, criterion, optimizer, scheduler, epoch, use_cuda,args)
        
        te_loss, te_top1, te_top5 = func.evaluate(testloader, model, criterion, epoch, use_cuda, args)

        # Logging loss, accuracy
        logger.append([epoch, tr_loss, tr_top1,te_loss, te_top1 ])
        
        
        #save model
        is_best = te_top1 > best_acc
        best_acc = max(te_top1, best_acc)
        print("Best Acc:%f"%(best_acc))
        save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'acc': te_top1,
                        'best_acc': best_acc,
                        'optimizer' : optimizer.state_dict(),
                        'scheduler' : scheduler.state_dict(),
                }, is_best, checkpoint=args.checkpoint)
        
    
    print('Best Test Err:%f'%(100-best_acc))
                

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


if __name__ == '__main__':
        main()
