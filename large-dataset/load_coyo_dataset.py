"""
Helper module to load and process Coyo-labeled-300m dataset.
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from torchvision import transforms
import warnings


class CoyoDataset(Dataset):
    """
    PyTorch Dataset wrapper for Coyo-labeled-300m.
    Converts multi-label to single-label by using the top-1 label.
    """
    
    def __init__(self, hf_dataset, transform=None, max_samples=None, num_classes=1000, 
                 use_top_k_classes=1000, cache_images=False):
        """
        Args:
            hf_dataset: Hugging Face dataset (from datasets library)
            transform: Image transforms
            max_samples: Maximum number of samples to use (None for all)
            num_classes: Number of classes to use (top-K classes by frequency)
            use_top_k_classes: Use top-K most frequent classes
            cache_images: Whether to cache downloaded images in memory
        """
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.cache_images = cache_images
        self.image_cache = {}
        
        # Filter and process dataset
        print(f"Processing Coyo-labeled-300m dataset...")
        try:
            dataset_size = len(hf_dataset)
            print(f"Total samples in dataset: {dataset_size}")
        except (TypeError, AttributeError):
            print("Dataset size unknown (streaming mode)")
        
        # Collect label frequencies to determine top classes
        label_counts = {}
        valid_samples = []
        
        print("Analyzing label distribution...")
        
        for idx, example in enumerate(hf_dataset):
            if max_samples and idx >= max_samples:
                break
            
            try:
                labels = example.get('labels', [])
                label_probs = example.get('label_probs', [])
                
                if labels and len(labels) > 0:
                    # Use top-1 label
                    top_label = labels[0]
                    if top_label < num_classes:  # Filter by num_classes if specified
                        label_counts[top_label] = label_counts.get(top_label, 0) + 1
                        valid_samples.append({
                            'top_label': top_label,
                            'url': example.get('url', ''),
                            'label_prob': label_probs[0] if label_probs else 0.0
                        })
            except Exception as e:
                warnings.warn(f"Error processing sample {idx}: {e}")
                continue
            
            if (idx + 1) % 10000 == 0:
                print(f"Processed {idx + 1} samples...")
        
        print(f"Found {len(valid_samples)} valid samples")
        
        # Select top-K classes by frequency
        if use_top_k_classes and use_top_k_classes < len(label_counts):
            sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
            top_classes = set([label for label, _ in sorted_labels[:use_top_k_classes]])
            valid_samples = [s for s in valid_samples if s['top_label'] in top_classes]
            print(f"Filtered to top {use_top_k_classes} classes: {len(valid_samples)} samples")
        
        # Remap labels to 0..num_classes-1
        unique_labels = sorted(set([s['top_label'] for s in valid_samples]))
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        
        self.samples = []
        for sample in valid_samples:
            self.samples.append({
                'label': label_mapping[sample['top_label']],
                'url': sample['url'],
                'label_prob': sample['label_prob']
            })
        
        self.num_classes = len(unique_labels)
        self.label_mapping = label_mapping
        self.reverse_label_mapping = {v: k for k, v in label_mapping.items()}
        
        print(f"Final dataset size: {len(self.samples)} samples")
        print(f"Number of classes: {self.num_classes}")
    
    def __len__(self):
        return len(self.samples)
    
    def _download_image(self, url):
        """Download image from URL"""
        if url in self.image_cache:
            return self.image_cache[url]
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
            if self.cache_images:
                self.image_cache[url] = image
            
            return image
        except Exception as e:
            warnings.warn(f"Failed to download image from {url}: {e}")
            # Return a black image as fallback
            return Image.new('RGB', (224, 224), color='black')
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Download and load image
        image = self._download_image(sample['url'])
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = sample['label']
        
        return image, label


def load_coyo_dataset(data_dir='./data', split='train', max_samples=None, 
                     num_classes=1000, use_top_k_classes=1000, 
                     cache_images=False, transform=None):
    """
    Load Coyo-labeled-300m dataset from Hugging Face.
    
    Args:
        data_dir: Directory for caching (not used for HF datasets)
        split: Dataset split ('train' or 'test')
        max_samples: Maximum number of samples to load (for testing)
        num_classes: Number of classes to use
        use_top_k_classes: Use top-K most frequent classes
        cache_images: Whether to cache images in memory
        transform: Image transforms
    
    Returns:
        dataset, num_classes, class_names
    """
    print(f"Loading Coyo-labeled-300m dataset from Hugging Face...")
    print("Note: This may take a while for the first time as metadata is downloaded.")
    
    # Load dataset from Hugging Face
    # Note: This loads metadata only, images are downloaded on-demand
    # For very large datasets, streaming mode is recommended
    print("Loading dataset from Hugging Face...")
    try:
        # Try non-streaming first (faster for smaller subsets)
        hf_dataset = load_dataset("kakaobrain/coyo-labeled-300m", split=split, streaming=False)
        print("Loaded in non-streaming mode")
    except Exception as e:
        print(f"Non-streaming mode failed: {e}")
        print("Switching to streaming mode (recommended for large datasets)...")
        hf_dataset = load_dataset("kakaobrain/coyo-labeled-300m", split=split, streaming=True)
        print("Loaded in streaming mode")
    
    # Create PyTorch dataset
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    dataset = CoyoDataset(
        hf_dataset, 
        transform=transform,
        max_samples=max_samples,
        num_classes=num_classes,
        use_top_k_classes=use_top_k_classes,
        cache_images=cache_images
    )
    
    # Generate class names (using ImageNet-21k class indices)
    class_names = [f"Class_{i}" for i in range(dataset.num_classes)]
    
    return dataset, dataset.num_classes, class_names


def create_train_test_split(coyo_train_dataset, test_split_ratio=0.1, random_seed=42):
    """
    Create train/test split from Coyo dataset.
    
    Args:
        coyo_train_dataset: CoyoDataset instance
        test_split_ratio: Ratio for test set
        random_seed: Random seed for reproducibility
    
    Returns:
        train_dataset, test_dataset
    """
    from torch.utils.data import Subset
    
    np.random.seed(random_seed)
    dataset_size = len(coyo_train_dataset)
    test_size = int(dataset_size * test_split_ratio)
    indices = np.random.permutation(dataset_size)
    
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    train_dataset = Subset(coyo_train_dataset, train_indices.tolist())
    test_dataset = Subset(coyo_train_dataset, test_indices.tolist())
    
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    
    return train_dataset, test_dataset
