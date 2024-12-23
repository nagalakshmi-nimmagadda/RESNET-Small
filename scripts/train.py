import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import logging
from src.model.resnet import ResNet50
from src.data.dataset import ImageNetDataset
from src.utils.metrics import AverageMeter, accuracy
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet50 on ImageNet')
    parser.add_argument('--data-dir', type=str, required=True,
                      help='path to ImageNet dataset')
    parser.add_argument('--epochs', type=int, default=100,
                      help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=256,
                      help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.1,
                      help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                      help='momentum for SGD')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                      help='weight decay')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = ResNet50().to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                         momentum=args.momentum,
                         weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Create datasets and dataloaders
    train_dataset = ImageNetDataset(args.data_dir, split='train')
    val_dataset = ImageNetDataset(args.data_dir, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=8, pin_memory=True)
    
    # Training loop
    best_acc = 0.0
    for epoch in range(args.epochs):
        # Train for one epoch
        train_one_epoch(train_loader, model, criterion, optimizer, epoch, device, logger)
        
        # Evaluate on validation set
        acc1 = validate(val_loader, model, criterion, device, logger)
        
        # Remember best accuracy and save checkpoint
        is_best = acc1 > best_acc
        best_acc = max(acc1, best_acc)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best)
        
        scheduler.step()
        
def train_one_epoch(train_loader, model, criterion, optimizer, epoch, device, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    model.train()
    end = time.time()
    
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        images = images.to(device)
        target = target.to(device)
        
        # Compute output
        output = model(images)
        loss = criterion(output, target)
        
        # Measure accuracy and record loss
        acc1 = accuracy(output, target)[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        
        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % 100 == 0:
            logger.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1))

if __name__ == '__main__':
    main() 