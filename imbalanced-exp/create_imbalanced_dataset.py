"""
Script to create imbalanced versions of CIFAR10 dataset.
Supports different imbalance strategies: long-tail, step, etc.
"""

import torch
from torch.utils.data import Dataset
from torchvision import datasets
import numpy as np


class ImbalancedCIFAR10(Dataset):
    """Wrapper to create imbalanced CIFAR10 dataset"""
    
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    
    def __len__(self):
        return len(self.indices)


def create_long_tail_imbalance(dataset, imbalance_ratio=0.1, num_classes=10):
    """
    Create long-tail imbalanced dataset.
    
    Args:
        dataset: CIFAR10 dataset
        imbalance_ratio: Ratio between smallest and largest class (e.g., 0.1 means 
                         smallest class has 10% of largest class samples)
        num_classes: Number of classes (10 for CIFAR10)
    
    Returns:
        Indices for imbalanced dataset
    """
    # Get all indices grouped by class
    class_indices = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    # Calculate samples per class using exponential decay
    max_samples = len(class_indices[0])
    samples_per_class = []
    for i in range(num_classes):
        num_samples = int(max_samples * (imbalance_ratio ** (i / (num_classes - 1))))
        samples_per_class.append(num_samples)
    
    # Sample indices for each class
    imbalanced_indices = []
    class_distribution = {}
    for i in range(num_classes):
        num_samples = min(samples_per_class[i], len(class_indices[i]))
        selected = np.random.choice(class_indices[i], num_samples, replace=False)
        imbalanced_indices.extend(selected)
        class_distribution[i] = len(selected)
    
    print("Class distribution (long-tail):")
    for cls, count in sorted(class_distribution.items()):
        print(f"  Class {cls}: {count} samples")
    
    return imbalanced_indices, class_distribution


def create_step_imbalance(dataset, imbalance_ratio=0.1, num_classes=10, minority_classes=None):
    """
    Create step imbalanced dataset where some classes are minority.
    
    Args:
        dataset: CIFAR10 dataset
        imbalance_ratio: Ratio for minority classes
        num_classes: Number of classes
        minority_classes: List of class indices to make minority (default: first half)
    
    Returns:
        Indices for imbalanced dataset
    """
    if minority_classes is None:
        minority_classes = list(range(num_classes // 2))
    
    # Get all indices grouped by class
    class_indices = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    # Calculate samples per class
    max_samples = len(class_indices[0])
    imbalanced_indices = []
    class_distribution = {}
    
    for i in range(num_classes):
        if i in minority_classes:
            num_samples = int(max_samples * imbalance_ratio)
        else:
            num_samples = max_samples
        
        num_samples = min(num_samples, len(class_indices[i]))
        selected = np.random.choice(class_indices[i], num_samples, replace=False)
        imbalanced_indices.extend(selected)
        class_distribution[i] = len(selected)
    
    print("Class distribution (step imbalance):")
    for cls, count in sorted(class_distribution.items()):
        status = "minority" if cls in minority_classes else "majority"
        print(f"  Class {cls} ({status}): {count} samples")
    
    return imbalanced_indices, class_distribution


if __name__ == '__main__':
    # Example usage
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    
    print("Creating long-tail imbalanced dataset...")
    indices, dist = create_long_tail_imbalance(dataset, imbalance_ratio=0.1)
    print(f"Total samples: {len(indices)}")
    
    print("\nCreating step imbalanced dataset...")
    indices2, dist2 = create_step_imbalance(dataset, imbalance_ratio=0.1, minority_classes=[0, 1, 2, 3, 4])
    print(f"Total samples: {len(indices2)}")
