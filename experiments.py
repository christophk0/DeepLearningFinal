import torch
import yaml
import os
import json
from datetime import datetime
import argparse
from pathlib import Path

from CNN import CNN
from VisionTransormer import VisionTransformer
from metrics import MetricsTracker, ExperimentLogger, count_parameters
from utils import (
    set_seed, get_device, load_dataset, create_sparse_dataset, 
    create_imbalanced_dataset, add_noise_corruption, create_data_loaders,
    print_dataset_info, save_checkpoint
)


class ExperimentRunner:
    """Main class for running systematic experiments"""
    
    def __init__(self, base_config_path='config.yaml', results_dir='results'):
        self.base_config = yaml.safe_load(open(base_config_path, 'r'))
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.results_dir / 'logs').mkdir(exist_ok=True)
        (self.results_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.results_dir / 'summaries').mkdir(exist_ok=True)
        
        self.logger = ExperimentLogger(str(self.results_dir / 'logs'))
        self.device = get_device()
        print(f"Using device: {self.device}")
        
    def create_model(self, model_type, config, num_classes=10):
        """Create model based on type and configuration"""
        if model_type == 'cnn':
            return CNN(config['cnn'], self.device, num_classes=num_classes)
        elif model_type == 'vt':
            return VisionTransformer(config['vision_transformer'], self.device, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train_epoch(self, model, train_loader, metrics_tracker):
        """Train for one epoch"""
        model.train()
        total_loss = 0
        total_accuracy = 0
        total_training_time = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            result = model.training_step(batch)
            
            # Handle both old and new return formats
            if isinstance(result, dict):
                total_loss += result['loss'].item()
                total_accuracy += result['accuracy']
                total_training_time += result['training_time']
            else:
                # Fallback for old format
                total_loss += result.item()
                # Calculate accuracy manually
                data, target = batch
                data, target = data.to(self.device), target.to(self.device)
                with torch.no_grad():
                    output = model(data)
                    pred = output.argmax(dim=1)
                    accuracy = (pred == target).float().mean().item()
                    total_accuracy += accuracy
            
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_training_time = total_training_time / num_batches if total_training_time > 0 else 0
        
        return avg_loss, avg_accuracy, avg_training_time
    
    def evaluate(self, model, test_loader):
        """Evaluate model on test set"""
        model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        total_inference_time = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                result = model.test_step(batch)
                
                # Handle both old and new return formats
                if isinstance(result, dict):
                    total_loss += result['loss']
                    total_correct += result['correct']
                    total_samples += result['total']
                    total_inference_time += result['inference_time']
                    all_predictions.extend(result['predictions'])
                    all_targets.extend(result['targets'])
                else:
                    # Fallback for old format
                    total_loss += result[0]
                    total_correct += result[1]
                    
                    # Get predictions and targets manually
                    data, target = batch
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    
                    all_predictions.extend(pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
                    total_samples += target.size(0)
        
        avg_loss = total_loss / len(test_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        avg_inference_time = total_inference_time / len(test_loader) if total_inference_time > 0 else 0
        
        return avg_loss, accuracy, all_predictions, all_targets, avg_inference_time
    
    def run_single_experiment(self, experiment_name, config, train_dataset, test_dataset, num_classes=10):
        """Run a single experiment with given configuration and datasets"""
        print(f"\n{'='*60}")
        print(f"Running experiment: {experiment_name}")
        print(f"{'='*60}")
        
        # Set seed for reproducibility
        set_seed(config.get('seed', 42))
        
        # Print dataset information
        print_dataset_info(train_dataset, "Training Dataset")
        print_dataset_info(test_dataset, "Test Dataset")
        
        # Create data loaders
        train_loader, test_loader = create_data_loaders(
            train_dataset, test_dataset, 
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 4)
        )
        
        # Create model
        model = self.create_model(config['model_type'], config, num_classes)
        print(f"\nModel: {config['model_type']}")
        print(f"Parameters: {count_parameters(model):,}")
        
        # Initialize metrics tracker
        metrics_tracker = MetricsTracker(num_classes, experiment_name)
        
        # Log experiment start
        self.logger.log_experiment_start(experiment_name, config)
        
        # Log model information
        model_info = model.get_model_info()
        print(f"\nModel Information:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        # Training loop
        metrics_tracker.start_training()
        num_epochs = config['num_epochs']
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            metrics_tracker.start_epoch()
            
            # Train
            train_loss, train_acc, train_time = self.train_epoch(model, train_loader, metrics_tracker)
            
            # Evaluate
            val_loss, val_acc, predictions, targets, inference_time = self.evaluate(model, test_loader)
            
            # Update metrics
            metrics_tracker.update_train_metrics(train_loss, train_acc)
            metrics_tracker.update_val_metrics(val_loss, val_acc, predictions, targets)
            metrics_tracker.end_epoch()
            
            # Log epoch results
            f1_score = metrics_tracker.val_f1_scores[-1]
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Time: {train_time:.4f}s")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, F1: {f1_score:.4f}, Inference Time: {inference_time:.4f}s")
            
            self.logger.log_epoch(experiment_name, epoch, train_loss, train_acc, val_loss, val_acc, f1_score)
            
            # Save checkpoint for best model
            if val_acc == metrics_tracker.best_val_acc:
                checkpoint_path = self.results_dir / 'checkpoints' / f'{experiment_name}_best.pth'
                save_checkpoint(model, model.optimizer, epoch, val_loss, checkpoint_path)
        
        # Get final results
        total_time = metrics_tracker.end_training()
        summary = metrics_tracker.get_summary()
        
        # Add model information to summary
        summary['model_info'] = model_info
        
        print(f"\nExperiment completed in {total_time:.2f} seconds")
        print(f"Best validation accuracy: {summary['best_val_accuracy']:.4f}")
        print(f"Total parameters: {model_info['total_parameters']:,}")
        print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
        
        # Save results
        metrics_tracker.save_detailed_results(str(self.results_dir / 'summaries'))
        self.logger.log_experiment_end(experiment_name, summary)
        
        return summary
    
    def run_baseline_experiments(self):
        """Run Experiment Set A - Baseline comparisons"""
        print("\n" + "="*80)
        print("RUNNING EXPERIMENT SET A - BASELINE COMPARISONS")
        print("="*80)
        
        results = {}
        
        # Load standard CIFAR-10
        train_dataset, test_dataset, num_classes = load_dataset('cifar10')
        
        # Experiment configurations
        experiments = [
            {
                'name': 'cnn_pretrained_cifar10',
                'config': {
                    **self.base_config,
                    'model_type': 'cnn',
                    'cnn': {**self.base_config['cnn'], 'pretrained': True, 'freeze_pretrained_layers': False}
                }
            },
            {
                'name': 'cnn_scratch_cifar10',
                'config': {
                    **self.base_config,
                    'model_type': 'cnn',
                    'cnn': {**self.base_config['cnn'], 'pretrained': False, 'freeze_pretrained_layers': False}
                }
            },
            {
                'name': 'vit_pretrained_cifar10',
                'config': {
                    **self.base_config,
                    'model_type': 'vt',
                    'vision_transformer': {**self.base_config['vision_transformer'], 'pretrained': True, 'freeze_pretrained_layers': False}
                }
            },
            {
                'name': 'vit_scratch_cifar10',
                'config': {
                    **self.base_config,
                    'model_type': 'vt',
                    'vision_transformer': {**self.base_config['vision_transformer'], 'pretrained': False, 'freeze_pretrained_layers': False}
                }
            }
        ]
        
        for exp in experiments:
            try:
                summary = self.run_single_experiment(
                    exp['name'], exp['config'], train_dataset, test_dataset, num_classes
                )
                results[exp['name']] = summary
            except Exception as e:
                print(f"Error in experiment {exp['name']}: {str(e)}")
                results[exp['name']] = {'error': str(e)}
        
        # Save combined results
        with open(self.results_dir / 'baseline_experiments_summary.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_data_regime_experiments(self):
        """Run Experiment Set B - Data regime comparisons"""
        print("\n" + "="*80)
        print("RUNNING EXPERIMENT SET B - DATA REGIME COMPARISONS")
        print("="*80)
        
        results = {}
        
        # Load base dataset
        base_train, test_dataset, num_classes = load_dataset('cifar10')
        
        # Create different data regimes
        data_regimes = [
            ('sparse_1pct', create_sparse_dataset(base_train, 0.01)),
            ('sparse_5pct', create_sparse_dataset(base_train, 0.05)),
            ('sparse_10pct', create_sparse_dataset(base_train, 0.10)),
            ('sparse_25pct', create_sparse_dataset(base_train, 0.25)),
            ('sparse_50pct', create_sparse_dataset(base_train, 0.50)),
            ('imbalanced', create_imbalanced_dataset(base_train, 0.1)),
            ('noisy_gaussian', add_noise_corruption(base_train, 'gaussian', 0.1)),
            ('noisy_blur', add_noise_corruption(base_train, 'blur', 0.1))
        ]
        
        # Test both CNN and ViT on each regime
        model_configs = [
            ('cnn', {**self.base_config, 'model_type': 'cnn'}),
            ('vit', {**self.base_config, 'model_type': 'vt'})
        ]
        
        for regime_name, train_dataset in data_regimes:
            for model_name, config in model_configs:
                exp_name = f"{model_name}_{regime_name}"
                try:
                    summary = self.run_single_experiment(
                        exp_name, config, train_dataset, test_dataset, num_classes
                    )
                    results[exp_name] = summary
                except Exception as e:
                    print(f"Error in experiment {exp_name}: {str(e)}")
                    results[exp_name] = {'error': str(e)}
        
        # Save results
        with open(self.results_dir / 'data_regime_experiments_summary.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_architectural_experiments(self):
        """Run Experiment Set C - Architectural variations"""
        print("\n" + "="*80)
        print("RUNNING EXPERIMENT SET C - ARCHITECTURAL VARIATIONS")
        print("="*80)
        
        results = {}
        
        # Load standard dataset
        train_dataset, test_dataset, num_classes = load_dataset('cifar10')
        
        # Architectural variations
        experiments = [
            {
                'name': 'cnn_shallow',
                'config': {
                    **self.base_config,
                    'model_type': 'cnn',
                    'cnn': {**self.base_config['cnn'], 'num_layers_to_drop': 3}
                }
            },
            {
                'name': 'vit_shallow',
                'config': {
                    **self.base_config,
                    'model_type': 'vt',
                    'vision_transformer': {**self.base_config['vision_transformer'], 'num_encoder_layers_to_drop': 8}
                }
            },
            # Note: Local attention ViT will be implemented in the next step
        ]
        
        for exp in experiments:
            try:
                summary = self.run_single_experiment(
                    exp['name'], exp['config'], train_dataset, test_dataset, num_classes
                )
                results[exp['name']] = summary
            except Exception as e:
                print(f"Error in experiment {exp['name']}: {str(e)}")
                results[exp['name']] = {'error': str(e)}
        
        # Save results
        with open(self.results_dir / 'architectural_experiments_summary.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Run CNN vs ViT experiments')
    parser.add_argument('--experiment_set', choices=['baseline', 'data_regime', 'architectural', 'all'], 
                       default='all', help='Which experiment set to run')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--results_dir', default='results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create experiment runner
    runner = ExperimentRunner(args.config, args.results_dir)
    
    # Run experiments
    if args.experiment_set in ['baseline', 'all']:
        runner.run_baseline_experiments()
    
    if args.experiment_set in ['data_regime', 'all']:
        runner.run_data_regime_experiments()
    
    if args.experiment_set in ['architectural', 'all']:
        runner.run_architectural_experiments()
    
    print(f"\nAll experiments completed! Results saved to {args.results_dir}")


if __name__ == '__main__':
    main()
