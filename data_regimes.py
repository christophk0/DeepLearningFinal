import torch
import numpy as np
import random
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset
from PIL import Image, ImageFilter
import torchvision.transforms.functional as TF
from utils import set_seed, get_class_distribution


class DataRegimeFactory:
    """Factory class for creating different data regime variations"""
    
    def __init__(self, base_dataset, seed=42):
        self.base_dataset = base_dataset
        self.seed = seed
        set_seed(seed)
        
    def create_sparse_dataset(self, fraction=0.1, maintain_balance=True):
        """
        Create a sparse version of the dataset with only a fraction of the data
        
        Args:
            fraction: Fraction of data to keep (0.0 to 1.0)
            maintain_balance: If True, maintain class balance; if False, sample randomly
        """
        set_seed(self.seed)
        
        if maintain_balance:
            return self._create_balanced_sparse_dataset(fraction)
        else:
            return self._create_random_sparse_dataset(fraction)
    
    def _create_balanced_sparse_dataset(self, fraction):
        """Create sparse dataset while maintaining class balance"""
        # Get targets
        if hasattr(self.base_dataset, 'targets'):
            targets = np.array(self.base_dataset.targets)
        else:
            targets = np.array([self.base_dataset[i][1] for i in range(len(self.base_dataset))])
        
        # Get indices for each class
        class_indices = {}
        for class_idx in np.unique(targets):
            class_indices[class_idx] = np.where(targets == class_idx)[0]
        
        # Sample from each class
        selected_indices = []
        for class_idx, indices in class_indices.items():
            n_samples = max(1, int(len(indices) * fraction))  # At least 1 sample per class
            selected = np.random.choice(indices, n_samples, replace=False)
            selected_indices.extend(selected)
        
        return Subset(self.base_dataset, selected_indices)
    
    def _create_random_sparse_dataset(self, fraction):
        """Create sparse dataset by random sampling"""
        n_samples = int(len(self.base_dataset) * fraction)
        indices = np.random.choice(len(self.base_dataset), n_samples, replace=False)
        return Subset(self.base_dataset, indices)
    
    def create_imbalanced_dataset(self, imbalance_type='step', imbalance_ratio=0.1, minority_classes=None):
        """
        Create an imbalanced version of the dataset
        
        Args:
            imbalance_type: 'step' (binary imbalance), 'exponential', or 'custom'
            imbalance_ratio: For step: fraction of data for minority classes
                           For exponential: decay factor
            minority_classes: List of classes to make minority (if None, use half)
        """
        set_seed(self.seed)
        
        if imbalance_type == 'step':
            return self._create_step_imbalanced_dataset(imbalance_ratio, minority_classes)
        elif imbalance_type == 'exponential':
            return self._create_exponential_imbalanced_dataset(imbalance_ratio)
        else:
            raise ValueError(f"Unknown imbalance type: {imbalance_type}")
    
    def _create_step_imbalanced_dataset(self, imbalance_ratio, minority_classes):
        """Create dataset with step imbalance (binary: majority vs minority classes)"""
        # Get targets
        if hasattr(self.base_dataset, 'targets'):
            targets = np.array(self.base_dataset.targets)
        else:
            targets = np.array([self.base_dataset[i][1] for i in range(len(self.base_dataset))])
        
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
        
        return Subset(self.base_dataset, selected_indices)
    
    def _create_exponential_imbalanced_dataset(self, decay_factor):
        """Create dataset with exponential imbalance"""
        # Get targets
        if hasattr(self.base_dataset, 'targets'):
            targets = np.array(self.base_dataset.targets)
        else:
            targets = np.array([self.base_dataset[i][1] for i in range(len(self.base_dataset))])
        
        unique_classes = sorted(np.unique(targets))
        
        selected_indices = []
        for i, class_idx in enumerate(unique_classes):
            class_indices = np.where(targets == class_idx)[0]
            
            # Exponential decay: each class has decay_factor^i of the samples
            fraction = decay_factor ** i
            n_samples = max(1, int(len(class_indices) * fraction))
            
            selected = np.random.choice(class_indices, n_samples, replace=False)
            selected_indices.extend(selected)
        
        return Subset(self.base_dataset, selected_indices)


class CorruptedDataset(Dataset):
    """Dataset wrapper that applies various types of corruption to images"""
    
    def __init__(self, base_dataset, corruption_type='gaussian_noise', severity=1, seed=42):
        self.base_dataset = base_dataset
        self.corruption_type = corruption_type
        self.severity = severity
        self.seed = seed
        set_seed(seed)
        
        # Define severity levels (1-5, where 5 is most severe)
        self.severity_levels = {
            'gaussian_noise': [0.08, 0.12, 0.18, 0.26, 0.38],
            'shot_noise': [60, 25, 12, 5, 3],
            'impulse_noise': [0.03, 0.06, 0.09, 0.17, 0.27],
            'defocus_blur': [0.5, 0.6, 0.7, 0.8, 1.0],
            'motion_blur': [3, 4, 5, 6, 8],
            'zoom_blur': [1.01, 1.02, 1.03, 1.04, 1.06],
            'brightness': [0.1, 0.2, 0.3, 0.4, 0.5],
            'contrast': [0.4, 0.3, 0.2, 0.1, 0.05],
            'elastic_transform': [244, 244*2, 244*3, 244*4, 244*5],
            'pixelate': [0.95, 0.9, 0.85, 0.8, 0.7],
            'jpeg_compression': [25, 18, 15, 10, 7]
        }
        
        if corruption_type not in self.severity_levels:
            raise ValueError(f"Unknown corruption type: {corruption_type}")
        
        self.corruption_param = self.severity_levels[corruption_type][severity - 1]
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        
        # Convert to PIL if it's a tensor
        if isinstance(image, torch.Tensor):
            # Denormalize if needed (assuming ImageNet normalization)
            if image.min() < 0:  # Likely normalized
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image = image * std + mean
                image = torch.clamp(image, 0, 1)
            
            image = TF.to_pil_image(image)
        
        # Apply corruption
        corrupted_image = self._apply_corruption(image)
        
        # Convert back to tensor and normalize
        corrupted_image = TF.to_tensor(corrupted_image)
        corrupted_image = TF.normalize(corrupted_image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        return corrupted_image, label
    
    def _apply_corruption(self, image):
        """Apply the specified corruption to the image"""
        if self.corruption_type == 'gaussian_noise':
            return self._gaussian_noise(image)
        elif self.corruption_type == 'shot_noise':
            return self._shot_noise(image)
        elif self.corruption_type == 'impulse_noise':
            return self._impulse_noise(image)
        elif self.corruption_type == 'defocus_blur':
            return self._defocus_blur(image)
        elif self.corruption_type == 'motion_blur':
            return self._motion_blur(image)
        elif self.corruption_type == 'brightness':
            return self._brightness(image)
        elif self.corruption_type == 'contrast':
            return self._contrast(image)
        elif self.corruption_type == 'pixelate':
            return self._pixelate(image)
        else:
            # Fallback to gaussian noise
            return self._gaussian_noise(image)
    
    def _gaussian_noise(self, image):
        """Add Gaussian noise"""
        image_array = np.array(image).astype(np.float32) / 255.0
        noise = np.random.normal(0, self.corruption_param, image_array.shape)
        noisy_image = image_array + noise
        noisy_image = np.clip(noisy_image, 0, 1)
        return Image.fromarray((noisy_image * 255).astype(np.uint8))
    
    def _shot_noise(self, image):
        """Add shot noise (Poisson noise)"""
        image_array = np.array(image).astype(np.float32)
        noisy_image = np.random.poisson(image_array * self.corruption_param) / self.corruption_param
        noisy_image = np.clip(noisy_image, 0, 255)
        return Image.fromarray(noisy_image.astype(np.uint8))
    
    def _impulse_noise(self, image):
        """Add impulse noise (salt and pepper)"""
        image_array = np.array(image)
        mask = np.random.random(image_array.shape[:2]) < self.corruption_param
        
        # Salt noise
        salt_mask = mask & (np.random.random(image_array.shape[:2]) < 0.5)
        image_array[salt_mask] = 255
        
        # Pepper noise
        pepper_mask = mask & ~salt_mask
        image_array[pepper_mask] = 0
        
        return Image.fromarray(image_array)
    
    def _defocus_blur(self, image):
        """Apply defocus blur"""
        radius = self.corruption_param
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def _motion_blur(self, image):
        """Apply motion blur (simplified as directional blur)"""
        # Create a motion blur kernel
        kernel_size = int(self.corruption_param)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size//2, :] = 1.0
        kernel = kernel / kernel_size
        
        # Apply convolution (simplified)
        return image.filter(ImageFilter.GaussianBlur(radius=kernel_size/4))
    
    def _brightness(self, image):
        """Adjust brightness"""
        enhancer = transforms.ColorJitter(brightness=self.corruption_param)
        return enhancer(image)
    
    def _contrast(self, image):
        """Adjust contrast"""
        enhancer = transforms.ColorJitter(contrast=self.corruption_param)
        return enhancer(image)
    
    def _pixelate(self, image):
        """Apply pixelation"""
        width, height = image.size
        new_width = int(width * self.corruption_param)
        new_height = int(height * self.corruption_param)
        
        # Downscale and upscale
        small_image = image.resize((new_width, new_height), Image.NEAREST)
        return small_image.resize((width, height), Image.NEAREST)


def create_data_regime_suite(base_dataset, seed=42):
    """
    Create a comprehensive suite of data regime variations
    
    Returns a dictionary of dataset variations
    """
    factory = DataRegimeFactory(base_dataset, seed)
    
    regime_suite = {}
    
    # Sparse datasets
    sparse_fractions = [0.01, 0.05, 0.10, 0.25, 0.50]
    for fraction in sparse_fractions:
        regime_suite[f'sparse_{int(fraction*100)}pct'] = factory.create_sparse_dataset(fraction)
    
    # Imbalanced datasets
    regime_suite['imbalanced_step_01'] = factory.create_imbalanced_dataset('step', 0.1)
    regime_suite['imbalanced_step_05'] = factory.create_imbalanced_dataset('step', 0.5)
    regime_suite['imbalanced_exp_08'] = factory.create_imbalanced_dataset('exponential', 0.8)
    regime_suite['imbalanced_exp_05'] = factory.create_imbalanced_dataset('exponential', 0.5)
    
    # Corrupted datasets
    corruption_types = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 
        'defocus_blur', 'motion_blur', 'brightness', 'contrast', 'pixelate'
    ]
    
    for corruption in corruption_types:
        for severity in [1, 3, 5]:  # Low, medium, high severity
            regime_name = f'corrupted_{corruption}_sev{severity}'
            regime_suite[regime_name] = CorruptedDataset(base_dataset, corruption, severity, seed)
    
    return regime_suite


def analyze_data_regime(dataset, regime_name="Dataset"):
    """Analyze and print information about a data regime"""
    print(f"\n{regime_name} Analysis:")
    print(f"  Total samples: {len(dataset)}")
    
    # Get class distribution
    class_dist = get_class_distribution(dataset)
    print(f"  Number of classes: {len(class_dist)}")
    
    if len(class_dist) > 1:
        counts = list(class_dist.values())
        min_count = min(counts)
        max_count = max(counts)
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        print(f"  Class distribution:")
        print(f"    Min samples per class: {min_count}")
        print(f"    Max samples per class: {max_count}")
        print(f"    Mean samples per class: {mean_count:.1f}")
        print(f"    Std samples per class: {std_count:.1f}")
        print(f"    Imbalance ratio (min/max): {min_count/max_count:.3f}")
        
        # Calculate Gini coefficient for imbalance measure
        gini = calculate_gini_coefficient(counts)
        print(f"    Gini coefficient: {gini:.3f}")


def calculate_gini_coefficient(counts):
    """Calculate Gini coefficient to measure class imbalance"""
    counts = sorted(counts)
    n = len(counts)
    cumsum = np.cumsum(counts)
    return (n + 1 - 2 * sum((n + 1 - i) * count for i, count in enumerate(counts, 1)) / cumsum[-1]) / n


if __name__ == '__main__':
    # Example usage
    from torchvision import datasets
    
    # Load CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    cifar10 = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    
    # Create data regime suite
    regime_suite = create_data_regime_suite(cifar10)
    
    # Analyze a few regimes
    for name, dataset in list(regime_suite.items())[:5]:
        analyze_data_regime(dataset, name)
