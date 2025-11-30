#!/usr/bin/env python3
"""
Quick test script to run a subset of experiments for demonstration
"""

import torch
import yaml
import os
from pathlib import Path

from CNN import CNN
from VisionTransormer import VisionTransformer
from metrics import MetricsTracker, ExperimentLogger
from utils import set_seed, get_device, load_dataset, create_data_loaders
from data_regimes import create_data_regime_suite


def run_quick_experiment(model_type='cnn', data_regime='standard', num_epochs=3):
    """Run a quick experiment for testing"""
    
    # Setup
    set_seed(42)
    device = get_device()
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Load config
    config = yaml.safe_load(open('config.yaml', 'r'))
    config['num_epochs'] = num_epochs  # Override for quick test
    config['batch_size'] = 32  # Smaller batch size for speed
    
    # Load dataset
    train_dataset, test_dataset, num_classes = load_dataset('cifar10')
    
    # Apply data regime
    if data_regime == 'sparse_10pct':
        from data_regimes import DataRegimeFactory
        factory = DataRegimeFactory(train_dataset)
        train_dataset = factory.create_sparse_dataset(0.1)
    elif data_regime == 'imbalanced':
        from data_regimes import DataRegimeFactory
        factory = DataRegimeFactory(train_dataset)
        train_dataset = factory.create_imbalanced_dataset('step', 0.1)
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(train_dataset, test_dataset, config['batch_size'])
    
    # Create model
    if model_type == 'cnn':
        model = CNN(config['cnn'], device, num_classes)
    else:
        model = VisionTransformer(config['vision_transformer'], device, num_classes)
    
    # Setup metrics
    experiment_name = f"{model_type}_{data_regime}_quick"
    metrics_tracker = MetricsTracker(num_classes, experiment_name)
    
    print(f"Running {experiment_name}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    metrics_tracker.start_training()
    
    for epoch in range(num_epochs):
        print(f"\\nEpoch {epoch+1}/{num_epochs}")
        metrics_tracker.start_epoch()
        
        # Train
        model.train()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            result = model.training_step(batch)
            if isinstance(result, dict):
                total_loss += result['loss'].item()
                total_accuracy += result['accuracy']
            else:
                total_loss += result.item()
                # Calculate accuracy manually
                data, target = batch
                data, target = data.to(device), target.to(device)
                with torch.no_grad():
                    output = model(data)
                    pred = output.argmax(dim=1)
                    accuracy = (pred == target).float().mean().item()
                    total_accuracy += accuracy
            
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {total_loss/(batch_idx+1):.4f}")
        
        train_loss = total_loss / num_batches
        train_acc = total_accuracy / num_batches
        
        # Evaluate
        model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                result = model.test_step(batch)
                if isinstance(result, dict):
                    total_loss += result['loss']
                    all_predictions.extend(result['predictions'])
                    all_targets.extend(result['targets'])
                else:
                    total_loss += result[0]
                    data, target = batch
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    all_predictions.extend(pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
        
        val_loss = total_loss / len(test_loader)
        val_acc = sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_targets)
        
        # Update metrics
        metrics_tracker.update_train_metrics(train_loss, train_acc)
        metrics_tracker.update_val_metrics(val_loss, val_acc, all_predictions, all_targets)
        metrics_tracker.end_epoch()
        
        # Log results
        f1_score = metrics_tracker.val_f1_scores[-1]
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, F1: {f1_score:.4f}")
    
    # Get final results
    total_time = metrics_tracker.end_training()
    summary = metrics_tracker.get_summary()
    
    print(f"\\nExperiment completed in {total_time:.2f} seconds")
    print(f"Best validation accuracy: {summary['best_val_accuracy']:.4f}")
    
    # Save results
    metrics_tracker.save_detailed_results(str(results_dir / 'summaries'))
    
    return summary


def main():
    """Run a few quick experiments"""
    experiments = [
        ('cnn', 'standard'),
        ('vt', 'standard'),
        ('cnn', 'sparse_10pct'),
        ('vt', 'sparse_10pct'),
    ]
    
    results = {}
    
    for model_type, data_regime in experiments:
        try:
            print(f"\\n{'='*60}")
            print(f"Running {model_type} on {data_regime} data")
            print(f"{'='*60}")
            
            summary = run_quick_experiment(model_type, data_regime, num_epochs=3)
            results[f"{model_type}_{data_regime}"] = summary
            
        except Exception as e:
            print(f"Error in {model_type}_{data_regime}: {str(e)}")
            results[f"{model_type}_{data_regime}"] = {'error': str(e)}
    
    # Print summary
    print(f"\\n{'='*60}")
    print("QUICK TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for exp_name, result in results.items():
        if 'error' in result:
            print(f"{exp_name}: ERROR - {result['error']}")
        else:
            print(f"{exp_name}: Acc={result['best_val_accuracy']:.4f}, F1={result['final_f1_score']:.4f}")
    
    return results


if __name__ == '__main__':
    results = main()
