"""
Evaluation metrics for large dataset experiments.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, test_loader, device, num_classes=100, class_names=None):
    """
    Evaluate model on test set and return overall metrics.
    For large datasets, only overall metrics are computed (no per-class breakdown).
    
    Args:
        model: Trained model
        test_loader: DataLoader for test set
        device: Device to run evaluation on
        num_classes: Number of classes (used for F1 calculation)
        class_names: Optional list of class names (not used, kept for compatibility)
    
    Returns:
        Dictionary with overall metrics
    """
    model.eval()
    
    # Initialize metric accumulators
    total_loss = torch.tensor(0.0).to(device)
    total_correct = torch.tensor(0.0).to(device)
    total_samples = torch.tensor(0.0).to(device)
    
    # For macro F1 calculation
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
            total_loss += loss.sum()
            
            # Get predictions
            pred = output.argmax(dim=1)
            correct = (pred == target).type(torch.float)
            
            # Accumulate overall accuracy
            total_correct += correct.sum()
            total_samples += torch.tensor(len(target), dtype=torch.float).to(device)
            
            # Store predictions and targets for F1 calculation
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Compute final metrics
    test_loss = (total_loss / total_samples).item()
    accuracy = (total_correct / total_samples).item()
    
    # Calculate F1 scores using sklearn for efficiency
    from sklearn.metrics import f1_score
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Macro F1 (unweighted mean of per-class F1)
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    # Weighted F1 (weighted by class frequency)
    f1_weighted = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    metrics = {
        'test_loss': test_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }
    
    return metrics


def print_metrics(metrics, model_name, class_names=None):
    """Print formatted overall metrics"""
    print(f"\n{'='*60}")
    print(f"Results for {model_name}")
    print(f"{'='*60}")
    print(f"Test Loss: {metrics['test_loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")


# Removed plot_confusion_matrix and plot_per_class_metrics functions
# For large datasets, we only use overall metrics
