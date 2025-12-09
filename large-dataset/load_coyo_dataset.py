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
        self.failed_downloads = 0  # Track failed image downloads
        self.total_downloads = 0  # Track total download attempts
        
        # Filter and process dataset
        print(f"Processing Coyo-labeled-300m dataset (max_samples={max_samples})...")
        try:
            dataset_size = len(hf_dataset)
            print(f"Total samples in dataset: {dataset_size}")
        except (TypeError, AttributeError):
            print("Dataset size unknown (streaming mode)")
        
        # Collect label frequencies to determine top classes
        label_counts = {}
        valid_samples = []
        invalid_count = 0
        
        print(f"Analyzing label distribution (max_samples={max_samples})...")
        
        # Debug: Check first few examples to understand dataset structure
        first_example_checked = False
        
        for idx, example in enumerate(hf_dataset):
            if max_samples and idx >= max_samples:
                print(f"Reached max_samples limit: {max_samples} raw samples processed")
                break
            
            # Debug: Print structure of first example
            if not first_example_checked:
                print(f"\nDEBUG: First example structure:")
                print(f"  Keys: {list(example.keys())}")
                print(f"  Sample data (first 200 chars): {str(example)[:200]}")
                # Try to find label-related fields
                for key in example.keys():
                    val = example[key]
                    if 'label' in key.lower() or 'class' in key.lower():
                        print(f"  {key}: {type(val)} = {val if not isinstance(val, (list, dict)) else f'{type(val).__name__} with {len(val)} items'}")
                first_example_checked = True
            
            try:
                # Try multiple possible field names for labels
                labels = None
                label_probs = None
                
                # Check various possible field names
                if 'labels' in example:
                    labels = example['labels']
                elif 'label' in example:
                    val = example['label']
                    labels = [val] if isinstance(val, (int, np.integer)) else (val if isinstance(val, (list, tuple, np.ndarray)) else None)
                elif 'class' in example:
                    val = example['class']
                    labels = [val] if isinstance(val, (int, np.integer)) else (val if isinstance(val, (list, tuple, np.ndarray)) else None)
                elif 'class_id' in example:
                    val = example['class_id']
                    labels = [val] if isinstance(val, (int, np.integer)) else (val if isinstance(val, (list, tuple, np.ndarray)) else None)
                elif 'category' in example:
                    val = example['category']
                    labels = [val] if isinstance(val, (int, np.integer)) else (val if isinstance(val, (list, tuple, np.ndarray)) else None)
                
                # Handle label_probs
                if 'label_probs' in example:
                    label_probs = example['label_probs']
                elif 'prob' in example:
                    label_probs = example['prob']
                elif 'probability' in example:
                    label_probs = example['probability']
                
                # Convert to list if needed
                if labels is not None:
                    if isinstance(labels, (int, np.integer)):
                        labels = [labels]
                    elif isinstance(labels, np.ndarray):
                        labels = labels.tolist()
                    elif not isinstance(labels, (list, tuple)):
                        labels = None
                
                if labels and len(labels) > 0:
                    # Use top-1 label
                    top_label = int(labels[0])
                    # Don't filter by num_classes during collection - we'll filter later
                    label_counts[top_label] = label_counts.get(top_label, 0) + 1
                    valid_samples.append({
                        'top_label': top_label,
                        'url': example.get('url', example.get('image_url', example.get('image', ''))),
                        'label_prob': float(label_probs[0]) if label_probs and len(label_probs) > 0 else 0.0
                    })
                else:
                    invalid_count += 1  # No labels found
            except Exception as e:
                invalid_count += 1
                if idx < 5:  # Only warn for first few errors
                    print(f"DEBUG: Error processing sample {idx}: {e}")
                    print(f"  Example keys: {list(example.keys()) if hasattr(example, 'keys') else 'N/A'}")
                continue
            
            if (idx + 1) % 10000 == 0:
                print(f"Processed {idx + 1} raw samples, found {len(valid_samples)} valid samples so far...")
        
        print(f"\nSample processing summary:")
        print(f"  Raw samples processed: {idx + 1 if 'idx' in locals() else 0}")
        print(f"  Valid samples found: {len(valid_samples)}")
        print(f"  Invalid samples (no labels): {invalid_count}")
        print(f"  Label distribution: {len(label_counts)} unique labels found")
        if label_counts:
            top_5_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  Top 5 labels by frequency: {top_5_labels}")
        
        if len(valid_samples) == 0:
            raise ValueError(
                f"\n❌ ERROR: No valid samples found after processing {idx + 1 if 'idx' in locals() else 0} raw samples!\n"
                f"This could mean:\n"
                f"1. The dataset structure is different than expected (check DEBUG output above)\n"
                f"2. The 'labels' field doesn't exist or is named differently\n"
                f"3. All samples were filtered out\n"
                f"4. Network issues preventing data access\n\n"
                f"Please check the DEBUG output above to see the actual dataset structure.\n"
                f"If you see the dataset keys, we may need to update the code to use the correct field names."
            )
        
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
        
        print(f"\n✓ Final dataset size: {len(self.samples)} samples")
        print(f"✓ Number of classes: {self.num_classes}")
        print(f"\nNote: Images are downloaded on-demand during training.")
        print(f"      Failed downloads will use black images as fallback, which may affect model performance.")
    
    def __len__(self):
        return len(self.samples)
    
    def _download_image(self, url):
        """Download image from URL"""
        if url in self.image_cache:
            return self.image_cache[url]
        
        self.total_downloads += 1
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
            if self.cache_images:
                self.image_cache[url] = image
            
            return image
        except Exception as e:
            self.failed_downloads += 1
            # Only warn occasionally to reduce noise (every 50th failure)
            if self.failed_downloads % 50 == 0:
                warnings.warn(
                    f"Image download failures: {self.failed_downloads}/{self.total_downloads} "
                    f"({100*self.failed_downloads/self.total_downloads:.1f}%). "
                    f"Using black images as fallback. This may affect model performance."
                )
            # Return a black image as fallback
            return Image.new('RGB', (224, 224), color='black')
    
    def get_download_stats(self):
        """Get statistics about image downloads"""
        if self.total_downloads == 0:
            return None
        success_rate = (self.total_downloads - self.failed_downloads) / self.total_downloads * 100
        return {
            'total_attempts': self.total_downloads,
            'failed': self.failed_downloads,
            'successful': self.total_downloads - self.failed_downloads,
            'success_rate': success_rate
        }
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Download and load image
        image = self._download_image(sample['url'])
        
        # Ensure image is RGB and has valid size
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                # If transform fails, create a black image and apply transform
                warnings.warn(f"Transform failed for sample {idx}, using black image: {e}")
                image = Image.new('RGB', (224, 224), color='black')
                image = self.transform(image)
        
        # Verify tensor shape (should be [3, 224, 224])
        if isinstance(image, torch.Tensor):
            if len(image.shape) == 3 and image.shape[0] == 3:
                # Check height and width
                h, w = image.shape[1], image.shape[2]
                if h != 224 or w != 224:
                    # Force resize if shape is wrong - convert to PIL, resize, then back to tensor
                    if self.total_downloads % 100 == 0:  # Only warn occasionally
                        warnings.warn(f"Image shape {image.shape} is incorrect, forcing resize to (3, 224, 224)")
                    from torchvision.transforms.functional import to_pil_image, resize, center_crop, to_tensor, normalize
                    # Convert tensor back to PIL
                    pil_img = to_pil_image(image)
                    # Resize and crop
                    pil_img = resize(pil_img, (256, 256))
                    pil_img = center_crop(pil_img, 224)
                    # Convert back to tensor and normalize
                    image = to_tensor(pil_img)
                    image = normalize(image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        
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
    # For small max_samples, use streaming to avoid downloading entire dataset
    print("Loading dataset from Hugging Face...")
    
    # Use streaming mode if max_samples is small (< 1M) to avoid downloading entire dataset
    use_streaming = max_samples is not None and max_samples < 1000000
    
    if use_streaming:
        print(f"Using streaming mode (max_samples={max_samples} is small, avoids full dataset download)...")
        hf_dataset = load_dataset("kakaobrain/coyo-labeled-300m", split=split, streaming=True)
        print("Loaded in streaming mode")
    else:
        try:
            # Try non-streaming first (faster for larger subsets)
            print("Using non-streaming mode (may download full dataset metadata first)...")
            hf_dataset = load_dataset("kakaobrain/coyo-labeled-300m", split=split, streaming=False)
            print("Loaded in non-streaming mode")
        except Exception as e:
            print(f"Non-streaming mode failed: {e}")
            print("Switching to streaming mode...")
            hf_dataset = load_dataset("kakaobrain/coyo-labeled-300m", split=split, streaming=True)
            print("Loaded in streaming mode")
    
    # Create PyTorch dataset
    if transform is None:
        transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),  # Ensure RGB
            transforms.Resize((256, 256)),  # Resize to exactly 256x256 (both dimensions)
            transforms.CenterCrop(224),  # Crop to exactly 224x224
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
