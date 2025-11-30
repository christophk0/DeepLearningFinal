import torch
import numpy as np
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import os
from collections import Counter


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def get_transforms(dataset_name='cifar10', augment=True):
    """Get appropriate transforms for different datasets"""
    if dataset_name.lower() in ['cifar10', 'cifar100']:
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(224),  # Resize for pretrained models
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            
        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
    else:  # ImageNet or other
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    return train_transform, test_transform


def load_dataset(dataset_name='cifar10', data_dir='./data', augment=True):
    """Load standard datasets"""
    train_transform, test_transform = get_transforms(dataset_name, augment)
    
    if dataset_name.lower() == 'cifar10':
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True, transform=train_transform, download=True
        )
        test_dataset = datasets.CIFAR10(
            root=data_dir, train=False, transform=test_transform, download=True
        )
        num_classes = 10
        
    elif dataset_name.lower() == 'cifar100':
        train_dataset = datasets.CIFAR100(
            root=data_dir, train=True, transform=train_transform, download=True
        )
        test_dataset = datasets.CIFAR100(
            root=data_dir, train=False, transform=test_transform, download=True
        )
        num_classes = 100
        
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    return train_dataset, test_dataset, num_classes


def create_sparse_dataset(dataset, fraction=0.1, seed=42):
    """Create a sparse version of the dataset with only a fraction of the data"""
    set_seed(seed)
    
    # Get indices for each class
    targets = np.array([dataset[i][1] for i in range(len(dataset))])
    class_indices = {}
    for class_idx in np.unique(targets):
        class_indices[class_idx] = np.where(targets == class_idx)[0]
    
    # Sample from each class to maintain class balance
    selected_indices = []
    for class_idx, indices in class_indices.items():
        n_samples = max(1, int(len(indices) * fraction))  # At least 1 sample per class
        selected = np.random.choice(indices, n_samples, replace=False)
        selected_indices.extend(selected)
    
    return Subset(dataset, selected_indices)


def create_imbalanced_dataset(dataset, imbalance_ratio=0.1, minority_classes=None, seed=42):
    """
    Create an imbalanced version of the dataset
    imbalance_ratio: fraction of data to keep for minority classes
    minority_classes: list of classes to make minority (if None, use half the classes)
    """
    set_seed(seed)
    
    # Get targets
    targets = np.array([dataset[i][1] for i in range(len(dataset))])
    unique_classes = np.unique(targets)
    
    if minority_classes is None:
        # Make half the classes minority
        minority_classes = unique_classes[:len(unique_classes)//2]
    
    # Get indices for each class
    selected_indices = []
    for class_idx in unique_classes:
        class_indices = np.where(targets == class_idx)[0]
        
        if class_idx in minority_classes:
            # Keep only a fraction for minority classes
            n_samples = max(1, int(len(class_indices) * imbalance_ratio))
        else:
            # Keep all samples for majority classes
            n_samples = len(class_indices)
            
        selected = np.random.choice(class_indices, n_samples, replace=False)
        selected_indices.extend(selected)
    
    return Subset(dataset, selected_indices)


def add_noise_corruption(dataset, noise_type='gaussian', noise_level=0.1, seed=42):
    """Add noise corruption to dataset"""
    set_seed(seed)
    
    class NoisyDataset(torch.utils.data.Dataset):
        def __init__(self, original_dataset, noise_type, noise_level):
            self.dataset = original_dataset
            self.noise_type = noise_type
            self.noise_level = noise_level
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            image, label = self.dataset[idx]
            
            if self.noise_type == 'gaussian':
                noise = torch.randn_like(image) * self.noise_level
                image = torch.clamp(image + noise, 0, 1)
            elif self.noise_type == 'salt_pepper':
                mask = torch.rand_like(image) < self.noise_level
                image = torch.where(mask, torch.randint_like(image, 0, 2).float(), image)
            elif self.noise_type == 'blur':
                # Simple blur by averaging with neighbors (approximate)
                if len(image.shape) == 3:  # C, H, W
                    kernel_size = max(1, int(self.noise_level * 10))
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    image = transforms.functional.gaussian_blur(image, kernel_size)
                        
            return image, label
    
    return NoisyDataset(dataset, noise_type, noise_level)


def get_class_distribution(dataset):
    """Get the class distribution of a dataset"""
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'targets'):
        # For Subset datasets
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        # Fallback: iterate through dataset
        targets = [dataset[i][1] for i in range(len(dataset))]
    
    return Counter(targets)


def print_dataset_info(dataset, name="Dataset"):
    """Print information about a dataset"""
    print(f"\n{name} Information:")
    print(f"  Total samples: {len(dataset)}")
    
    class_dist = get_class_distribution(dataset)
    print(f"  Number of classes: {len(class_dist)}")
    print(f"  Class distribution: {dict(class_dist)}")
    
    # Calculate imbalance ratio
    if len(class_dist) > 1:
        min_count = min(class_dist.values())
        max_count = max(class_dist.values())
        imbalance_ratio = min_count / max_count
        print(f"  Imbalance ratio (min/max): {imbalance_ratio:.3f}")


def create_data_loaders(train_dataset, test_dataset, batch_size=64, num_workers=4):
    """Create data loaders with appropriate settings"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, test_loader


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']
