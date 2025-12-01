# Imbalanced Datasets Guide

This guide explains how to use existing imbalanced datasets and create custom ones for experiments.

## Available Imbalanced Datasets

### 1. CIFAR-10-LT (Long-Tailed CIFAR-10)

Pre-existing long-tailed version of CIFAR-10 from Hugging Face.

**Source:** [tomas-gajarsky/cifar10-lt](https://huggingface.co/datasets/tomas-gajarsky/cifar10-lt)

**Imbalance Factors Available:**
- `10`: Mild imbalance
- `50`: Moderate imbalance  
- `100`: Strong imbalance (default)
- `200`: Very strong imbalance

**Usage:**
```python
from load_imbalanced_datasets import load_cifar10_lt
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_data = load_cifar10_lt(imbalance_factor=100, split='train', transform=transform)
test_data = load_cifar10_lt(imbalance_factor=100, split='test', transform=transform)
```

### 2. CIFAR-100-LT (Long-Tailed CIFAR-100)

Pre-existing long-tailed version of CIFAR-100 from Hugging Face.

**Source:** [tomas-gajarsky/cifar100-lt](https://huggingface.co/datasets/tomas-gajarsky/cifar100-lt)

**Imbalance Factors Available:**
- `10`, `50`, `100`, `200`

**Usage:**
```python
from load_imbalanced_datasets import load_cifar100_lt

train_data = load_cifar100_lt(imbalance_factor=100, split='train', transform=transform)
test_data = load_cifar100_lt(imbalance_factor=100, split='test', transform=transform)
```

**Note:** For CIFAR-100, you'll need to modify `CNN.py` and `VisionTransormer.py` to output 100 classes instead of 10.

### 3. Custom Imbalanced Datasets

Create your own imbalanced versions of CIFAR-10/100 using the provided utilities.

**Long-tail imbalance:**
```python
from create_imbalanced_dataset import create_long_tail_imbalance, ImbalancedCIFAR10
from torchvision import datasets

full_train = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_indices, class_dist = create_long_tail_imbalance(full_train, imbalance_ratio=0.1)
train_dataset = ImbalancedCIFAR10(full_train, train_indices)
```

**Step imbalance:**
```python
from create_imbalanced_dataset import create_step_imbalance

train_indices, class_dist = create_step_imbalance(
    full_train, 
    imbalance_ratio=0.1, 
    minority_classes=[0, 1, 2, 3, 4]  # First 5 classes are minority
)
train_dataset = ImbalancedCIFAR10(full_train, train_indices)
```

## Comparison: Existing vs Custom Datasets

| Feature | Existing LT Datasets | Custom Imbalanced |
|---------|---------------------|-------------------|
| **Setup** | Simple download | Requires code execution |
| **Standardization** | Benchmark standard | Custom ratios |
| **Imbalance Factors** | Fixed (10, 50, 100, 200) | Any ratio (0.01-1.0) |
| **Reproducibility** | High (same for everyone) | Medium (depends on seed) |
| **Flexibility** | Limited to available factors | Full control |

## Recommended Workflow

1. **Start with existing datasets** (CIFAR-10-LT, CIFAR-100-LT) for:
   - Benchmark comparisons
   - Reproducible experiments
   - Standard imbalance factors

2. **Use custom datasets** for:
   - Specific imbalance ratios not available
   - Step imbalance patterns
   - Research into different imbalance types

## Installation

For existing datasets, install Hugging Face datasets:

```bash
pip install datasets huggingface_hub
```

## Example: Complete Experiment with CIFAR-10-LT

```python
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from load_imbalanced_datasets import load_cifar10_lt, get_cifar10_class_names
from CNN import CNN
from VisionTransormer import VisionTransformer
from evaluate import evaluate_model
import yaml

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load existing imbalanced dataset
train_data = load_cifar10_lt(imbalance_factor=100, split='train', transform=transform)
test_data = load_cifar10_lt(imbalance_factor=100, split='test', transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Load config and initialize models
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

cnn_model = CNN(config=config['cnn'], device=device)
vit_model = VisionTransformer(config=config['vision_transformer'], device=device)

# Train and evaluate...
```

## Troubleshooting

### Issue: Cannot load from Hugging Face

**Solution:** 
- Check internet connection
- Install: `pip install datasets huggingface_hub`
- Try different imbalance factor (10, 50, 100, 200)

### Issue: Dataset format errors

**Solution:**
- The `HuggingFaceCIFAR` wrapper handles different image formats
- If issues persist, check the dataset structure: `print(dataset[0])`

### Issue: CIFAR-100 requires 100 classes

**Solution:**
- Modify `CNN.py`: Change `nn.LazyLinear(10)` to `nn.LazyLinear(100)`
- Modify `VisionTransormer.py`: Change `nn.LazyLinear(10)` to `nn.LazyLinear(100)`

## References

- [CIFAR-10-LT on Hugging Face](https://huggingface.co/datasets/tomas-gajarsky/cifar10-lt)
- [CIFAR-100-LT on Hugging Face](https://huggingface.co/datasets/tomas-gajarsky/cifar100-lt)
- [Long-Tailed Recognition Paper](https://arxiv.org/abs/2004.11860)
