import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image

class ImageNetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, subset_size=None):
        """
        Args:
            root_dir: Root directory of ImageNet dataset
            split: 'train' or 'val'
            transform: Optional transform to be applied
            subset_size: If not None, use only this many classes
        """
        self.root_dir = os.path.join(root_dir, split)
        self.split = split
        self.transform = transform or self._get_default_transforms()
        
        # Get all classes
        self.classes = sorted(os.listdir(self.root_dir))
        if subset_size:
            self.classes = self.classes[:subset_size]
            
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _get_default_transforms(self):
        if self.split == 'train':
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])

    def _make_dataset(self):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                samples.append((os.path.join(class_dir, img_name), class_idx))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label 