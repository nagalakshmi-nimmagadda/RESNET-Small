import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import logging
import os
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
    parser.add_argument('--subset-size', type=int, default=None,
                      help='number of classes to use (None for full dataset)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                      help='directory to save checkpoints and logs')
    return parser.parse_args()

def validate(val_loader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    with torch.no_grad():
        for images, target in val_loader:
            images = images.to(device)
            target = target.to(device)
            
            output = model(images)
            loss = criterion(output, target)
            
            acc1 = accuracy(output, target)[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
    
    return losses.avg, top1.avg

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        filename=os.path.join(args.output_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Setup tensorboard
    writer = SummaryWriter(args.output_dir)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    num_classes = args.subset_size if args.subset_size else 1000
    model = ResNet50(num_classes=num_classes).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                         momentum=args.momentum,
                         weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Create datasets and dataloaders
    train_dataset = ImageNetDataset(args.data_dir, split='train', subset_size=args.subset_size)
    val_dataset = ImageNetDataset(args.data_dir, split='val', subset_size=args.subset_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=8, pin_memory=True)
    
    # Training loop
    best_acc = 0.0
    for epoch in range(args.epochs):
        # Train for one epoch
        model.train()
        for i, (images, target) in enumerate(train_loader):
            images = images.to(device)
            target = target.to(device)
            
            # Compute output and loss
            output = model(images)
            loss = criterion(output, target)
            
            # Compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log training progress
            if i % 100 == 0:
                acc1 = accuracy(output, target)[0]
                logger.info(
                    f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                    f'Loss {loss.item():.4f}\t'
                    f'Acc@1 {acc1.item():.3f}'
                )
                writer.add_scalar('training/loss', loss.item(), epoch * len(train_loader) + i)
                writer.add_scalar('training/accuracy', acc1.item(), epoch * len(train_loader) + i)
        
        # Evaluate on validation set
        val_loss, val_acc = validate(val_loader, model, criterion, device)
        logger.info(f'Validation: Loss {val_loss:.4f}\tAcc@1 {val_acc:.3f}')
        writer.add_scalar('validation/loss', val_loss, epoch)
        writer.add_scalar('validation/accuracy', val_acc, epoch)
        
        # Remember best accuracy and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc,
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.output_dir, 'checkpoint.pth'))
        
        if is_best:
            torch.save({
                'state_dict': model.state_dict(),
                'best_acc1': best_acc,
            }, os.path.join(args.output_dir, 'model_best.pth'))
        
        scheduler.step()
    
    writer.close()

if __name__ == '__main__':
    main() 