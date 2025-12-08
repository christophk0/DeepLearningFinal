"""
Main script to run large dataset experiments comparing CNN and Vision Transformer.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import yaml
import json
import os
import sys
from datetime import datetime

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CNN import CNN
from VisionTransormer import VisionTransformer
from evaluate import evaluate_model, print_metrics
from load_coyo_dataset import load_coyo_dataset, create_train_test_split


def train_model(model, train_loader, val_loader, num_epochs, device, print_freq=10):
    """Train model and return training history with validation loss"""
    history = {'loss': [], 'val_loss': [], 'epoch': []}
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_losses = []
        for batch_idx, batch in enumerate(train_loader):
            loss = model.training_step(batch)
            epoch_losses.append(loss.item())
            
            if batch_idx % print_freq == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        history['loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                data, target = batch
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_losses.append(loss.item())
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        history['val_loss'].append(avg_val_loss)
        history['epoch'].append(epoch)
        print(f"Epoch {epoch} completed. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return history


def plot_training_curves(history_cnn, history_vit, save_path):
    """Plot training and validation loss curves for both models"""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Training loss
    ax1.plot(history_cnn['epoch'], history_cnn['loss'], label='CNN Train', marker='o', linestyle='-')
    ax1.plot(history_vit['epoch'], history_vit['loss'], label='ViT Train', marker='s', linestyle='-')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Curves - Large Dataset')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Validation loss
    ax2.plot(history_cnn['epoch'], history_cnn['val_loss'], label='CNN Val', marker='o', linestyle='--')
    ax2.plot(history_vit['epoch'], history_vit['val_loss'], label='ViT Val', marker='s', linestyle='--')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss Curves - Large Dataset')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def load_dataset(dataset_name, data_dir='./data', max_samples=None, num_classes=None):
    """
    Load large dataset.
    
    Args:
        dataset_name: Name of dataset ('cifar100' or 'coyo')
        data_dir: Directory to store dataset
        max_samples: Maximum samples to load (for Coyo dataset)
        num_classes: Number of classes to use (for Coyo dataset)
    
    Returns:
        train_dataset, test_dataset, num_classes, class_names
    """
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    if dataset_name.lower() == 'cifar100':
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, transform=transform, download=True)
        num_classes = 100
        
        # CIFAR-100 class names
        class_names = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
            'worm'
        ]
        
        return train_dataset, test_dataset, num_classes, class_names
    
    elif dataset_name.lower() in ['coyo', 'coyo-labeled-300m', 'coyo300m']:
        # Load Coyo-labeled-300m dataset
        # Default: use top 1000 classes, max 100k samples for training (adjustable)
        if max_samples is None:
            max_samples = 100000  # Default limit for training
        if num_classes is None:
            num_classes = 1000  # Default: top 1000 classes
        
        print(f"Loading Coyo-labeled-300m dataset (max_samples={max_samples}, num_classes={num_classes})...")
        print("Note: This may take a while. Images are downloaded on-demand.")
        
        # Load training data
        train_coyo_dataset, num_classes_actual, class_names = load_coyo_dataset(
            data_dir=data_dir,
            split='train',
            max_samples=max_samples,
            num_classes=num_classes,
            use_top_k_classes=num_classes,
            cache_images=False,  # Don't cache to save memory
            transform=transform
        )
        
        # Create train/test split (80/20)
        train_dataset, test_dataset = create_train_test_split(
            train_coyo_dataset, 
            test_split_ratio=0.2
        )
        
        return train_dataset, test_dataset, num_classes_actual, class_names
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: 'cifar100', 'coyo'")


def run_experiment(config_path='../config.yaml', 
                   dataset='cifar100',
                   num_epochs=10,
                   output_dir='./results',
                   max_samples=None,
                   num_classes=None):
    """
    Run large dataset experiment.
    
    Args:
        config_path: Path to config.yaml
        dataset: Dataset name ('cifar100')
        num_epochs: Number of training epochs
        output_dir: Directory to save results
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(output_dir, f"large_dataset_{dataset}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create charts subdirectory
    charts_dir = os.path.join(exp_dir, 'charts')
    os.makedirs(charts_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    )
    print(f"Using device: {device}")
    
    # Load config
    config = yaml.safe_load(open(config_path, 'r'))
    
    # Load dataset
    print(f"\nLoading {dataset} dataset...")
    train_dataset, test_dataset, num_classes, class_names = load_dataset(
        dataset, 
        max_samples=max_samples,
        num_classes=num_classes
    )
    print(f"Dataset loaded: {num_classes} classes")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Validate dataset sizes before creating loaders
    if len(train_dataset) == 0:
        raise ValueError(
            f"❌ ERROR: Training dataset has 0 samples! Cannot proceed with training.\n"
            f"This likely means no valid samples were found during dataset processing.\n"
            f"Please check the dataset loading output above for details."
        )
    if len(test_dataset) == 0:
        raise ValueError(
            f"❌ ERROR: Test dataset has 0 samples! Cannot proceed with evaluation.\n"
            f"This likely means no valid samples were found during dataset processing.\n"
            f"Please check the dataset loading output above for details."
        )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize models with correct number of classes
    print("\n" + "="*60)
    print("Initializing CNN (ResNet)")
    print("="*60)
    cnn_config = config['cnn'].copy()
    
    # Create CNN model and update final layer
    cnn_model = CNN(config=cnn_config, device=device)
    # Update final layer for correct number of classes
    # First, get the feature dimension by doing a forward pass through the resnet backbone
    cnn_model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        features = cnn_model.resnet(dummy_input)
        feature_dim = features.shape[1]
    # Replace final layer
    cnn_model.final_layer = torch.nn.Linear(feature_dim, num_classes).to(device)
    # Update optimizer to include new final layer
    if cnn_model.freeze_pretrained_layers:
        cnn_model.optimizer = torch.optim.Adam(cnn_model.final_layer.parameters(), lr=cnn_config['learning_rate'])
    else:
        cnn_model.optimizer = torch.optim.Adam(cnn_model.parameters(), lr=cnn_config['learning_rate'])
    print(f"CNN final layer updated: {feature_dim} -> {num_classes} classes")
    
    print("\n" + "="*60)
    print("Initializing Vision Transformer")
    print("="*60)
    vit_config = config['vision_transformer'].copy()
    
    # Create ViT model and update final layer
    vit_model = VisionTransformer(config=vit_config, device=device)
    # Update final layer for correct number of classes
    vit_model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        features = vit_model.vit(dummy_input)
        feature_dim = features.shape[1]
    # Replace final layer
    vit_model.final_layer = torch.nn.Linear(feature_dim, num_classes).to(device)
    # Update optimizer to include new final layer
    if vit_model.freeze_pretrained_layers:
        vit_model.optimizer = torch.optim.Adam(vit_model.final_layer.parameters(), lr=vit_config['learning_rate'])
    else:
        vit_model.optimizer = torch.optim.Adam(vit_model.parameters(), lr=vit_config['learning_rate'])
    print(f"ViT final layer updated: {feature_dim} -> {num_classes} classes")
    
    # Train CNN
    print("\n" + "="*60)
    print("Training CNN...")
    print("="*60)
    cnn_history = train_model(cnn_model, train_loader, test_loader, num_epochs, device, 
                             print_freq=config.get('print_batch_frequency', 10))
    
    # Train ViT
    print("\n" + "="*60)
    print("Training ViT...")
    print("="*60)
    vit_history = train_model(vit_model, train_loader, test_loader, num_epochs, device,
                             print_freq=config.get('print_batch_frequency', 10))
    
    # Print download statistics if available (for Coyo dataset)
    if dataset.lower() in ['coyo', 'coyo-labeled-300m', 'coyo300m']:
        try:
            # Access the underlying CoyoDataset if it's wrapped in a Subset
            underlying_dataset = train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset
            if hasattr(underlying_dataset, 'get_download_stats'):
                stats = underlying_dataset.get_download_stats()
                if stats and stats['total_attempts'] > 0:
                    print("\n" + "="*60)
                    print("Image Download Statistics")
                    print("="*60)
                    print(f"Total download attempts: {stats['total_attempts']}")
                    print(f"Successful downloads: {stats['successful']}")
                    print(f"Failed downloads: {stats['failed']}")
                    print(f"Success rate: {stats['success_rate']:.2f}%")
                    print(f"\n⚠️  WARNING: {stats['failed']} images failed to download and were replaced with black images.")
                    print(f"   This may significantly affect model performance, especially with {stats['success_rate']:.1f}% success rate.")
        except Exception:
            pass  # Silently ignore if stats not available
    
    # Evaluate models
    print("\n" + "="*60)
    print("Evaluating CNN")
    print("="*60)
    cnn_metrics = evaluate_model(cnn_model, test_loader, device, num_classes=num_classes, class_names=class_names)
    print_metrics(cnn_metrics, "CNN (ResNet)", class_names)
    
    print("\n" + "="*60)
    print("Evaluating Vision Transformer")
    print("="*60)
    vit_metrics = evaluate_model(vit_model, test_loader, device, num_classes=num_classes, class_names=class_names)
    print_metrics(vit_metrics, "Vision Transformer", class_names)
    
    # Save results
    results = {
        'experiment_config': {
            'dataset': dataset,
            'num_classes': num_classes,
            'num_epochs': num_epochs,
            'train_samples': len(train_dataset),
            'test_samples': len(test_dataset),
            'max_samples': max_samples,
            'num_classes_config': num_classes
        },
        'cnn_metrics': {
            'test_loss': float(cnn_metrics['test_loss']),
            'accuracy': float(cnn_metrics['accuracy']),
            'f1_macro': float(cnn_metrics['f1_macro']),
            'f1_weighted': float(cnn_metrics['f1_weighted'])
        },
        'vit_metrics': {
            'test_loss': float(vit_metrics['test_loss']),
            'accuracy': float(vit_metrics['accuracy']),
            'f1_macro': float(vit_metrics['f1_macro']),
            'f1_weighted': float(vit_metrics['f1_weighted'])
        },
        'training_history': {
            'cnn_loss': [float(x) for x in cnn_history['loss']],
            'cnn_val_loss': [float(x) for x in cnn_history['val_loss']],
            'vit_loss': [float(x) for x in vit_history['loss']],
            'vit_val_loss': [float(x) for x in vit_history['val_loss']]
        }
    }
    
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    print("\nGenerating visualizations...")
    print(f"Saving charts to: {charts_dir}")
    
    # For large datasets, we only generate training curves
    # Per-class metrics and confusion matrices are skipped for efficiency
    plot_training_curves(
        cnn_history, vit_history,
        os.path.join(charts_dir, 'training_curves.png')
    )
    
    print(f"Training curves saved to: {charts_dir}")
    
    # Summary comparison
    print("\n" + "="*60)
    print("Summary Comparison")
    print("="*60)
    print(f"{'Metric':<25} {'CNN':<15} {'ViT':<15}")
    print("-" * 60)
    print(f"{'Accuracy':<25} {cnn_metrics['accuracy']*100:>6.2f}%      {vit_metrics['accuracy']*100:>6.2f}%")
    print(f"{'F1-Score (Macro)':<25} {cnn_metrics['f1_macro']:>6.4f}      {vit_metrics['f1_macro']:>6.4f}")
    print(f"{'F1-Score (Weighted)':<25} {cnn_metrics['f1_weighted']:>6.4f}      {vit_metrics['f1_weighted']:>6.4f}")
    print(f"{'Test Loss':<25} {cnn_metrics['test_loss']:>6.4f}      {vit_metrics['test_loss']:>6.4f}")
    
    print(f"\nAll results saved to: {exp_dir}")
    
    return results, exp_dir


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run large dataset experiments')
    parser.add_argument('--config', type=str, default='../config.yaml',
                       help='Path to config.yaml')
    parser.add_argument('--dataset', type=str, default='cifar100',
                       choices=['cifar100', 'coyo', 'coyo-labeled-300m', 'coyo300m'],
                       help='Dataset to use: cifar100 or coyo (Coyo-labeled-300m)')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to load (for Coyo dataset, default: 100000)')
    parser.add_argument('--num_classes', type=int, default=None,
                       help='Number of classes to use (for Coyo dataset, default: 1000)')
    
    args = parser.parse_args()
    
    run_experiment(
        config_path=args.config,
        dataset=args.dataset,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        num_classes=args.num_classes
    )
