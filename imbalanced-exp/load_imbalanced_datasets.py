"""
Helper functions to load existing imbalanced datasets.
Supports CIFAR-10-LT and CIFAR-100-LT from Hugging Face.
"""

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: Hugging Face datasets not available. Install with: pip install datasets")


class HuggingFaceCIFAR(Dataset):
    """Wrapper for Hugging Face CIFAR datasets"""
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Handle different image formats from Hugging Face
        image = item.get('img') or item.get('image')
        
        if isinstance(image, dict):
            # Convert dict to numpy array then to PIL
            image = Image.fromarray(np.array(image))
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            # Try to convert to PIL Image
            try:
                image = Image.fromarray(np.array(image))
            except:
                raise ValueError(f"Unknown image format: {type(image)}")
        
        if self.transform:
            image = self.transform(image)
        
        label = item['label']
        return image, label


def load_cifar10_lt(imbalance_factor=100, split='train', transform=None, root='./data'):
    """
    Load CIFAR-10-LT (Long-Tailed CIFAR-10) dataset from Hugging Face.
    
    Args:
        imbalance_factor: Imbalance factor (10, 50, 100, 200). Higher = more imbalanced.
        split: 'train' or 'test'
        transform: Torchvision transforms
        root: Root directory for caching
    
    Returns:
        Dataset object or None if loading fails
    """
    if not HF_AVAILABLE:
        print("Hugging Face datasets not available. Install with: pip install datasets")
        return None
    
    try:
        dataset_name = f"tomas-gajarsky/cifar10-lt-{imbalance_factor}"
        print(f"Loading {dataset_name} ({split})...")
        hf_dataset = load_dataset(dataset_name, split=split, cache_dir=root)
        return HuggingFaceCIFAR(hf_dataset, transform=transform)
    except Exception as e:
        print(f"Error loading CIFAR-10-LT: {e}")
        print("Available imbalance factors: 10, 50, 100, 200")
        return None


def load_cifar100_lt(imbalance_factor=100, split='train', transform=None, root='./data'):
    """
    Load CIFAR-100-LT (Long-Tailed CIFAR-100) dataset from Hugging Face.
    
    Args:
        imbalance_factor: Imbalance factor (10, 50, 100, 200). Higher = more imbalanced.
        split: 'train' or 'test'
        transform: Torchvision transforms
        root: Root directory for caching
    
    Returns:
        Dataset object or None if loading fails
    """
    if not HF_AVAILABLE:
        print("Hugging Face datasets not available. Install with: pip install datasets")
        return None
    
    try:
        dataset_name = f"tomas-gajarsky/cifar100-lt-{imbalance_factor}"
        print(f"Loading {dataset_name} ({split})...")
        hf_dataset = load_dataset(dataset_name, split=split, cache_dir=root)
        return HuggingFaceCIFAR(hf_dataset, transform=transform)
    except Exception as e:
        print(f"Error loading CIFAR-100-LT: {e}")
        print("Available imbalance factors: 10, 50, 100, 200")
        return None


def get_cifar10_class_names():
    """Get CIFAR-10 class names"""
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck']


def get_cifar100_class_names():
    """Get CIFAR-100 class names (coarse labels)"""
    # CIFAR-100 has 100 fine classes grouped into 20 coarse classes
    # For simplicity, return generic names
    return [f'class_{i}' for i in range(100)]


if __name__ == '__main__':
    # Example usage
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    print("Testing CIFAR-10-LT loading...")
    train_data = load_cifar10_lt(imbalance_factor=100, split='train', transform=transform)
    if train_data:
        print(f"✓ Loaded {len(train_data)} training samples")
        
        # Check class distribution
        from collections import Counter
        labels = [train_data[i][1] for i in range(min(1000, len(train_data)))]
        dist = Counter(labels)
        print(f"Sample class distribution (first 1000 samples):")
        for cls in sorted(dist.keys()):
            print(f"  Class {cls}: {dist[cls]} samples")
    else:
        print("✗ Failed to load dataset")
    
    print("\nTesting CIFAR-100-LT loading...")
    train_data_100 = load_cifar100_lt(imbalance_factor=100, split='train', transform=transform)
    if train_data_100:
        print(f"✓ Loaded {len(train_data_100)} training samples")
    else:
        print("✗ Failed to load dataset")
