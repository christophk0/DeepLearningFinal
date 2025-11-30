import torch
import time
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
from collections import defaultdict
import json
import os
from datetime import datetime


class MetricsTracker:
    """Track comprehensive metrics during training and evaluation"""
    
    def __init__(self, num_classes=10, experiment_name="experiment"):
        self.num_classes = num_classes
        self.experiment_name = experiment_name
        self.reset()
        
    def reset(self):
        """Reset all metrics for a new experiment"""
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []
        self.epoch_times = []
        self.per_class_metrics = []
        self.confusion_matrices = []
        self.start_time = None
        self.convergence_epoch = None
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
    def start_epoch(self):
        """Mark the start of an epoch"""
        self.epoch_start_time = time.time()
        
    def end_epoch(self):
        """Mark the end of an epoch and record time"""
        if hasattr(self, 'epoch_start_time'):
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
            
    def start_training(self):
        """Mark the start of training"""
        self.start_time = time.time()
        
    def end_training(self):
        """Mark the end of training and return total time"""
        if self.start_time:
            return time.time() - self.start_time
        return 0
        
    def update_train_metrics(self, loss, accuracy):
        """Update training metrics for current epoch"""
        self.train_losses.append(loss)
        self.train_accuracies.append(accuracy)
        
    def update_val_metrics(self, loss, accuracy, predictions, targets):
        """Update validation metrics for current epoch"""
        self.val_losses.append(loss)
        self.val_accuracies.append(accuracy)
        
        # Calculate F1 score
        f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
        self.val_f1_scores.append(f1)
        
        # Calculate per-class metrics
        precision, recall, f1_per_class, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        per_class_dict = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1_per_class.tolist(),
            'support': support.tolist()
        }
        self.per_class_metrics.append(per_class_dict)
        
        # Calculate confusion matrix
        cm = confusion_matrix(targets, predictions)
        self.confusion_matrices.append(cm.tolist())
        
        # Check for convergence (early stopping)
        if accuracy > self.best_val_acc:
            self.best_val_acc = accuracy
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        # Mark convergence if we haven't improved in 5 epochs
        if self.convergence_epoch is None and self.patience_counter >= 5:
            self.convergence_epoch = len(self.val_accuracies) - 5
            
    def get_convergence_info(self):
        """Get information about convergence"""
        if self.convergence_epoch is not None:
            return {
                'converged': True,
                'convergence_epoch': self.convergence_epoch,
                'convergence_accuracy': self.val_accuracies[self.convergence_epoch] if self.convergence_epoch < len(self.val_accuracies) else None
            }
        return {
            'converged': False,
            'convergence_epoch': None,
            'convergence_accuracy': None
        }
        
    def get_summary(self):
        """Get a summary of all metrics"""
        total_training_time = self.end_training() if self.start_time else 0
        convergence_info = self.get_convergence_info()
        
        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'total_epochs': len(self.train_losses),
            'total_training_time': total_training_time,
            'avg_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_train_accuracy': self.train_accuracies[-1] if self.train_accuracies else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'final_val_accuracy': self.val_accuracies[-1] if self.val_accuracies else None,
            'best_val_accuracy': self.best_val_acc,
            'final_f1_score': self.val_f1_scores[-1] if self.val_f1_scores else None,
            'convergence_info': convergence_info,
            'per_class_final': self.per_class_metrics[-1] if self.per_class_metrics else None
        }
        
        return summary
        
    def save_detailed_results(self, save_dir):
        """Save detailed results to files"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save summary
        summary = self.get_summary()
        with open(os.path.join(save_dir, f'{self.experiment_name}_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Save detailed metrics
        detailed = {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_f1_scores': self.val_f1_scores,
            'epoch_times': self.epoch_times,
            'per_class_metrics': self.per_class_metrics,
            'confusion_matrices': self.confusion_matrices
        }
        
        with open(os.path.join(save_dir, f'{self.experiment_name}_detailed.json'), 'w') as f:
            json.dump(detailed, f, indent=2)
            
        return summary


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_flops(model, input_shape=(1, 3, 224, 224)):
    """
    Estimate FLOPs for a model (simplified calculation)
    This is a basic implementation - for more accurate FLOP counting,
    consider using libraries like ptflops or fvcore
    """
    try:
        from ptflops import get_model_complexity_info
        macs, params = get_model_complexity_info(model, input_shape[1:], print_per_layer_stat=False)
        # MACs (multiply-accumulate operations) ≈ FLOPs/2
        flops = 2 * float(macs.replace('GMac', '').replace('MMac', '').replace('KMac', ''))
        if 'GMac' in macs:
            flops *= 1e9
        elif 'MMac' in macs:
            flops *= 1e6
        elif 'KMac' in macs:
            flops *= 1e3
        return int(flops)
    except ImportError:
        # Fallback: rough estimation based on parameters
        return count_parameters(model) * 2  # Very rough approximation
        

class ExperimentLogger:
    """Logger for experiment tracking"""
    
    def __init__(self, log_dir="results/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
    def log_experiment_start(self, experiment_name, config):
        """Log the start of an experiment"""
        log_file = os.path.join(self.log_dir, f"{experiment_name}.log")
        with open(log_file, 'w') as f:
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Start time: {datetime.now().isoformat()}\n")
            f.write(f"Configuration:\n{json.dumps(config, indent=2)}\n")
            f.write("-" * 50 + "\n")
            
    def log_epoch(self, experiment_name, epoch, train_loss, train_acc, val_loss, val_acc, f1_score):
        """Log epoch results"""
        log_file = os.path.join(self.log_dir, f"{experiment_name}.log")
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, F1: {f1_score:.4f}\n")
            
    def log_experiment_end(self, experiment_name, summary):
        """Log the end of an experiment"""
        log_file = os.path.join(self.log_dir, f"{experiment_name}.log")
        with open(log_file, 'a') as f:
            f.write("-" * 50 + "\n")
            f.write(f"Experiment completed at: {datetime.now().isoformat()}\n")
            f.write(f"Final results: {json.dumps(summary, indent=2)}\n")
