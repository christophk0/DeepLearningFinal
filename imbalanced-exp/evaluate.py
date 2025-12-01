"""
Evaluation metrics for imbalanced dataset experiments.
Uses AccumTensor from cka/metrics.py for metric accumulation.
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path to import cka metrics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cka.metrics import AccumTensor

import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, test_loader, device, num_classes=10, class_names=None):
    """
    Evaluate model on test set and return comprehensive metrics.
    Uses AccumTensor for metric accumulation following CKA experiment pattern.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test set
        device: Device to run evaluation on
        num_classes: Number of classes
        class_names: Optional list of class names
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    
    # Initialize AccumTensor metrics
    total_loss = AccumTensor(torch.tensor(0.0).to(device))
    total_correct = AccumTensor(torch.tensor(0.0).to(device))
    total_samples = AccumTensor(torch.tensor(0.0).to(device))
    
    # Per-class metrics
    per_class_correct = AccumTensor(torch.zeros(num_classes).to(device))
    per_class_total = AccumTensor(torch.zeros(num_classes).to(device))
    
    # For confusion matrix
    all_preds = []
    all_targets = []
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in test_loader:
            data, target = batch
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            # Accumulate loss
            total_loss.update(loss.sum())
            
            # Get predictions
            pred = output.argmax(dim=1)
            correct = (pred == target).type(torch.float)
            
            # Accumulate overall accuracy
            total_correct.update(correct.sum())
            total_samples.update(torch.tensor(len(target), dtype=torch.float).to(device))
            
            # Accumulate per-class metrics
            per_class_correct_batch = torch.zeros(num_classes).to(device)
            per_class_total_batch = torch.zeros(num_classes).to(device)
            for c in range(num_classes):
                class_mask = (target == c)
                if class_mask.any():
                    per_class_correct_batch[c] = (pred[class_mask] == target[class_mask]).type(torch.float).sum()
                    per_class_total_batch[c] = class_mask.type(torch.float).sum()
            per_class_correct.update(per_class_correct_batch)
            per_class_total.update(per_class_total_batch)
            
            # Store for confusion matrix
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Compute final metrics
    test_loss = (total_loss.compute() / total_samples.compute()).item()
    accuracy = (total_correct.compute() / total_samples.compute()).item()
    
    per_class_correct_tensor = per_class_correct.compute()
    per_class_total_tensor = per_class_total.compute()
    per_class_recall = (per_class_correct_tensor / (per_class_total_tensor + 1e-8)).cpu().numpy()
    
    # Calculate confusion matrix and derived metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Build confusion matrix manually
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(len(all_targets)):
        cm[all_targets[i], all_preds[i]] += 1
    
    # Per-class precision
    per_class_precision = np.diag(cm) / (cm.sum(axis=0) + 1e-8)
    
    # Per-class F1
    per_class_f1 = 2 * (per_class_precision * per_class_recall) / (per_class_precision + per_class_recall + 1e-8)
    
    # Macro and weighted F1
    f1_macro = per_class_f1.mean()
    class_weights = per_class_total_tensor.cpu().numpy()
    f1_weighted = (per_class_f1 * class_weights).sum() / (class_weights.sum() + 1e-8)
    
    metrics = {
        'test_loss': test_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': per_class_f1,
        'per_class_recall': per_class_recall,
        'per_class_precision': per_class_precision,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'targets': all_targets
    }
    
    return metrics


def print_metrics(metrics, model_name, class_names=None):
    """Print formatted metrics"""
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(metrics['f1_per_class']))]
    
    print(f"\n{'='*60}")
    print(f"Results for {model_name}")
    print(f"{'='*60}")
    print(f"Test Loss: {metrics['test_loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<15} {'Recall':<10} {'Precision':<12} {'F1-Score':<10}")
    print("-" * 50)
    for i, name in enumerate(class_names):
        print(f"{name:<15} {metrics['per_class_recall'][i]:<10.4f} "
              f"{metrics['per_class_precision'][i]:<12.4f} "
              f"{metrics['f1_per_class'][i]:<10.4f}")


def plot_confusion_matrix(cm, class_names, save_path, model_name):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_per_class_metrics(metrics_cnn, metrics_vit, class_names, save_path):
    """Plot comparison of per-class metrics between CNN and ViT"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    x = np.arange(len(class_names))
    width = 0.35
    
    # Per-class Recall
    axes[0].bar(x - width/2, metrics_cnn['per_class_recall'], width, label='CNN', alpha=0.8)
    axes[0].bar(x + width/2, metrics_vit['per_class_recall'], width, label='ViT', alpha=0.8)
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Recall')
    axes[0].set_title('Per-Class Recall')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Per-class Precision
    axes[1].bar(x - width/2, metrics_cnn['per_class_precision'], width, label='CNN', alpha=0.8)
    axes[1].bar(x + width/2, metrics_vit['per_class_precision'], width, label='ViT', alpha=0.8)
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Per-Class Precision')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    # Per-class F1-Score
    axes[2].bar(x - width/2, metrics_cnn['f1_per_class'], width, label='CNN', alpha=0.8)
    axes[2].bar(x + width/2, metrics_vit['f1_per_class'], width, label='ViT', alpha=0.8)
    axes[2].set_xlabel('Class')
    axes[2].set_ylabel('F1-Score')
    axes[2].set_title('Per-Class F1-Score')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(class_names, rotation=45, ha='right')
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-class metrics comparison saved to {save_path}")
