import os
import shutil
from pathlib import Path

def create_imagenet_subset(
    source_dir,
    target_dir,
    n_classes=100,  # Using 100 classes instead of 1000
    n_images_per_class=50  # 50 images per class for quick testing
):
    """Create a small subset of ImageNet for testing."""
    for split in ['train', 'val']:
        source_split = os.path.join(source_dir, split)
        target_split = os.path.join(target_dir, split)
        
        # Create target directory
        os.makedirs(target_split, exist_ok=True)
        
        # Get list of classes
        classes = sorted(os.listdir(source_split))[:n_classes]
        
        for class_name in classes:
            # Create class directory
            source_class = os.path.join(source_split, class_name)
            target_class = os.path.join(target_split, class_name)
            os.makedirs(target_class, exist_ok=True)
            
            # Copy subset of images
            images = sorted(os.listdir(source_class))[:n_images_per_class]
            for img in images:
                shutil.copy2(
                    os.path.join(source_class, img),
                    os.path.join(target_class, img)
                )
    
    print(f"Created subset with {n_classes} classes and {n_images_per_class} images per class")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to full ImageNet")
    parser.add_argument("--target", required=True, help="Path for subset")
    args = parser.parse_args()
    
    create_imagenet_subset(args.source, args.target) 