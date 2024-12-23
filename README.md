# ResNet50 ImageNet Training Project

This project implements training of ResNet50 from scratch on the ImageNet dataset using AWS EC2. The implementation achieves state-of-the-art performance with a target of 70% top-1 accuracy.

## Project Structure 
resnet50-imagenet/
├── src/
│ ├── model/
│ │ ├── blocks.py
│ │ └── resnet.py
│ ├── data/
│ │ └── dataset.py
│ └── utils/
│ └── metrics.py
├── scripts/
│ └── train.py
├── app/
│ └── gradio_app.py
├── requirements.txt
└── README.md


## Detailed Component Explanation

### 1. Model Implementation (`src/model/`)

#### `blocks.py`
This file contains the fundamental building blocks of ResNet50:

- **BasicBlock**: A basic residual block with two 3x3 convolutions
  - Input → Conv3x3 → BN → ReLU → Conv3x3 → BN → Add Input → ReLU
  - Used in smaller ResNet variants (ResNet18, ResNet34)

- **Bottleneck**: The main building block for ResNet50
  - Input → Conv1x1 → BN → ReLU → Conv3x3 → BN → ReLU → Conv1x1 → BN → Add Input → ReLU
  - Uses 1x1 convolutions to reduce and then expand dimensions
  - Has an expansion factor of 4 for the output channels

#### `resnet.py`
Implements the complete ResNet50 architecture:

- **Initial Layer**: 7x7 Conv with stride 2, followed by BatchNorm, ReLU, and MaxPool
- **Layer Structure**:
  - Layer1: 3 bottleneck blocks (64 channels)
  - Layer2: 4 bottleneck blocks (128 channels)
  - Layer3: 6 bottleneck blocks (256 channels)
  - Layer4: 3 bottleneck blocks (512 channels)
- **Final Layers**: Global Average Pooling followed by a fully connected layer
- **Weight Initialization**: Uses Kaiming initialization for better training convergence

### 2. Data Handling (`src/data/`)

#### `dataset.py`
Implements the ImageNet dataset loader:

- **ImageNetDataset Class**:
  - Handles both training and validation splits
  - Implements data augmentation:
    - Training: RandomResizedCrop, RandomHorizontalFlip, ColorJitter
    - Validation: Resize, CenterCrop
  - Normalizes images using ImageNet statistics:
    - Mean: [0.485, 0.456, 0.406]
    - Std: [0.229, 0.224, 0.225]

### 3. Training Script (`scripts/`)

#### `train.py`
Main training script with the following features:

- **Command Line Arguments**:
  - `--data-dir`: Path to ImageNet dataset
  - `--epochs`: Number of training epochs (default: 100)
  - `--batch-size`: Batch size (default: 256)
  - `--lr`: Learning rate (default: 0.1)
  - `--momentum`: SGD momentum (default: 0.9)
  - `--weight-decay`: Weight decay (default: 1e-4)

- **Training Features**:
  - Uses SGD optimizer with momentum
  - Implements CosineAnnealingLR scheduler
  - Tracks and saves best model checkpoints
  - Logs training metrics every 100 iterations

### 4. Deployment (`app/`)

#### `gradio_app.py`
Implements a user-friendly web interface for model inference:

- Loads trained model
- Preprocesses input images
- Returns top-5 predictions with probabilities
- Provides example images for testing


### 5. Requirements (`requirements.txt`)
Key dependencies:

orch>=1.8.0 # PyTorch deep learning framework
torchvision>=0.9.0 # Computer vision utilities
pillow>=8.0.0 # Image processing
numpy>=1.19.2 # Numerical computations
gradio>=2.0.0 # Web interface
tqdm>=4.50.0 # Progress bars


## Training on AWS EC2

### Instance Requirements
- Instance Type: p3.2xlarge (or better)
- GPU: NVIDIA V100
- Storage: At least 1TB EBS volume for ImageNet dataset

### Setup Instructions
1. Launch EC2 instance with Deep Learning AMI
2. Connect to instance:
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-instance
   ```
3. Clone repository and install requirements:
   ```bash
   git clone <repository-url>
   cd resnet50-imagenet
   pip install -r requirements.txt
   ```
4. Mount EBS volume for dataset:
   ```bash
   sudo mkfs -t ext4 /dev/xvdf
   sudo mkdir /data
   sudo mount /dev/xvdf /data
   ```
5. Start training:
   ```bash
   python scripts/train.py --data-dir /data/imagenet
   ```

## Model Performance
- Target Top-1 Accuracy: 70%
- Training Time: ~3-4 days on p3.2xlarge
- Final Model Size: ~98MB

## Hugging Face Integration
The trained model is deployed on Hugging Face Spaces:
- Live demo: [Your Hugging Face Space URL]
- Supports direct image upload and inference
- Returns top-5 class predictions with confidence scores

## References
1. Original ResNet Paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
2. ImageNet Dataset: [ImageNet Large Scale Visual Recognition Challenge](https://www.image-net.org/)
3. Training Optimizations: [DAWNBench](https://dawn.cs.stanford.edu/benchmark/)