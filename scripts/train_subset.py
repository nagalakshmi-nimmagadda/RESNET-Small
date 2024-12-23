import torch
import torch.nn as nn
from src.model.resnet import ResNet50
from src.data.dataset import ImageNetDataset
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to ImageNet subset")
    parser.add_argument("--epochs", type=int, default=10)  # Fewer epochs for testing
    parser.add_argument("--batch-size", type=int, default=32)  # Smaller batch size
    parser.add_argument("--num-classes", type=int, default=100)  # Subset classes
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Modified ResNet50 for fewer classes
    model = ResNet50(num_classes=args.num_classes)
    
    # Rest of the training code...
    # (Similar to train.py but with smaller parameters)

if __name__ == "__main__":
    main() 