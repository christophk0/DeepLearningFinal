"""
Main script to run imbalanced dataset experiments comparing CNN and Vision Transformer.
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
from create_imbalanced_dataset import ImbalancedCIFAR10, create_long_tail_imbalance, create_step_imbalance
from evaluate import evaluate_model, print_metrics, plot_confusion_matrix, plot_per_class_metrics


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
    ax1.set_title('Training Loss Curves - Imbalanced Dataset')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Validation loss
    ax2.plot(history_cnn['epoch'], history_cnn['val_loss'], label='CNN Val', marker='o', linestyle='--')
    ax2.plot(history_vit['epoch'], history_vit['val_loss'], label='ViT Val', marker='s', linestyle='--')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss Curves - Imbalanced Dataset')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def run_experiment(config_path='../config.yaml', 
                   imbalance_type='long_tail',
                   imbalance_ratio=0.1,
                   num_epochs=10,
                   output_dir='./results'):
    """
    Run imbalanced dataset experiment.
    
    Args:
        config_path: Path to config.yaml
        imbalance_type: 'long_tail' or 'step'
        imbalance_ratio: Ratio for imbalanced dataset
        num_epochs: Number of training epochs
        output_dir: Directory to save results
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(output_dir, f"imbalanced_{imbalance_type}_{imbalance_ratio}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    )
    print(f"Using device: {device}")
    
    # Load config
    config = yaml.safe_load(open(config_path, 'r'))
    
    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Load full CIFAR10 datasets
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    
    # Create imbalanced training dataset
    print(f"\nCreating {imbalance_type} imbalanced dataset with ratio {imbalance_ratio}...")
    if imbalance_type == 'long_tail':
        train_indices, class_distribution = create_long_tail_imbalance(
            train_dataset, imbalance_ratio=imbalance_ratio
        )
    elif imbalance_type == 'step':
        train_indices, class_distribution = create_step_imbalance(
            train_dataset, imbalance_ratio=imbalance_ratio
        )
    else:
        raise ValueError(f"Unknown imbalance_type: {imbalance_type}")
    
    # Create imbalanced dataset wrapper
    imbalanced_train = ImbalancedCIFAR10(train_dataset, train_indices)
    train_loader = DataLoader(imbalanced_train, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Save class distribution
    with open(os.path.join(exp_dir, 'class_distribution.json'), 'w') as f:
        json.dump({str(k): v for k, v in class_distribution.items()}, f, indent=2)
    
    # CIFAR10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Initialize models
    print("\n" + "="*60)
    print("Training CNN (ResNet)")
    print("="*60)
    cnn_config = config['cnn'].copy()
    cnn_model = CNN(config=cnn_config, device=device)
    
    print("\n" + "="*60)
    print("Training Vision Transformer")
    print("="*60)
    vit_config = config['vision_transformer'].copy()
    vit_model = VisionTransformer(config=vit_config, device=device)
    
    # Train CNN
    print("\nTraining CNN...")
    cnn_history = train_model(cnn_model, train_loader, test_loader, num_epochs, device, 
                             print_freq=config.get('print_batch_frequency', 10))
    
    # Train ViT
    print("\nTraining ViT...")
    vit_history = train_model(vit_model, train_loader, test_loader, num_epochs, device,
                             print_freq=config.get('print_batch_frequency', 10))
    
    # Evaluate models
    print("\n" + "="*60)
    print("Evaluating CNN")
    print("="*60)
    cnn_metrics = evaluate_model(cnn_model, test_loader, device, num_classes=10, class_names=class_names)
    print_metrics(cnn_metrics, "CNN (ResNet)", class_names)
    
    print("\n" + "="*60)
    print("Evaluating Vision Transformer")
    print("="*60)
    vit_metrics = evaluate_model(vit_model, test_loader, device, num_classes=10, class_names=class_names)
    print_metrics(vit_metrics, "Vision Transformer", class_names)
    
    # Save results
    results = {
        'experiment_config': {
            'imbalance_type': imbalance_type,
            'imbalance_ratio': imbalance_ratio,
            'num_epochs': num_epochs,
            'class_distribution': {str(k): v for k, v in class_distribution.items()}
        },
        'cnn_metrics': {
            'test_loss': float(cnn_metrics['test_loss']),
            'accuracy': float(cnn_metrics['accuracy']),
            'f1_macro': float(cnn_metrics['f1_macro']),
            'f1_weighted': float(cnn_metrics['f1_weighted']),
            'f1_per_class': [float(x) for x in cnn_metrics['f1_per_class']],
            'per_class_recall': [float(x) for x in cnn_metrics['per_class_recall']],
            'per_class_precision': [float(x) for x in cnn_metrics['per_class_precision']]
        },
        'vit_metrics': {
            'test_loss': float(vit_metrics['test_loss']),
            'accuracy': float(vit_metrics['accuracy']),
            'f1_macro': float(vit_metrics['f1_macro']),
            'f1_weighted': float(vit_metrics['f1_weighted']),
            'f1_per_class': [float(x) for x in vit_metrics['f1_per_class']],
            'per_class_recall': [float(x) for x in vit_metrics['per_class_recall']],
            'per_class_precision': [float(x) for x in vit_metrics['per_class_precision']]
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
    
    # Confusion matrices
    plot_confusion_matrix(
        cnn_metrics['confusion_matrix'], 
        class_names,
        os.path.join(exp_dir, 'confusion_matrix_cnn.png'),
        'CNN (ResNet)'
    )
    
    plot_confusion_matrix(
        vit_metrics['confusion_matrix'],
        class_names,
        os.path.join(exp_dir, 'confusion_matrix_vit.png'),
        'Vision Transformer'
    )
    
    # Per-class metrics comparison
    plot_per_class_metrics(
        cnn_metrics, vit_metrics, class_names,
        os.path.join(exp_dir, 'per_class_metrics_comparison.png')
    )
    
    # Training curves
    plot_training_curves(
        cnn_history, vit_history,
        os.path.join(exp_dir, 'training_curves.png')
    )
    
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
    
    parser = argparse.ArgumentParser(description='Run imbalanced dataset experiments')
    parser.add_argument('--config', type=str, default='../config.yaml',
                       help='Path to config.yaml')
    parser.add_argument('--imbalance_type', type=str, default='long_tail',
                       choices=['long_tail', 'step'],
                       help='Type of imbalance: long_tail or step')
    parser.add_argument('--imbalance_ratio', type=float, default=0.1,
                       help='Imbalance ratio (0.1 means smallest class has 10%% of largest)')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    run_experiment(
        config_path=args.config,
        imbalance_type=args.imbalance_type,
        imbalance_ratio=args.imbalance_ratio,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir
    )
