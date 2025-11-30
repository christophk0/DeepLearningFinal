import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ExperimentVisualizer:
    """Class for creating visualizations from experiment results"""
    
    def __init__(self, results_dir='results'):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)
        
    def load_experiment_results(self, experiment_pattern="*_summary.json"):
        """Load all experiment results matching the pattern"""
        summary_files = list((self.results_dir / 'summaries').glob(experiment_pattern))
        results = {}
        
        for file_path in summary_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                experiment_name = file_path.stem.replace('_summary', '')
                results[experiment_name] = data
                
        return results
    
    def load_detailed_results(self, experiment_name):
        """Load detailed results for a specific experiment"""
        detailed_file = self.results_dir / 'summaries' / f'{experiment_name}_detailed.json'
        if detailed_file.exists():
            with open(detailed_file, 'r') as f:
                return json.load(f)
        return None
    
    def plot_training_curves(self, experiment_names, save_path=None):
        """Plot training and validation curves for multiple experiments"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Curves Comparison', fontsize=16, fontweight='bold')
        
        for exp_name in experiment_names:
            detailed = self.load_detailed_results(exp_name)
            if detailed is None:
                continue
                
            epochs = range(1, len(detailed['train_losses']) + 1)
            
            # Training and validation loss
            axes[0, 0].plot(epochs, detailed['train_losses'], label=f'{exp_name} (train)', alpha=0.7)
            axes[0, 0].plot(epochs, detailed['val_losses'], label=f'{exp_name} (val)', alpha=0.7, linestyle='--')
            
            # Training and validation accuracy
            axes[0, 1].plot(epochs, detailed['train_accuracies'], label=f'{exp_name} (train)', alpha=0.7)
            axes[0, 1].plot(epochs, detailed['val_accuracies'], label=f'{exp_name} (val)', alpha=0.7, linestyle='--')
            
            # F1 scores
            axes[1, 0].plot(epochs, detailed['val_f1_scores'], label=f'{exp_name}', alpha=0.7)
            
            # Epoch times
            axes[1, 1].plot(epochs, detailed['epoch_times'], label=f'{exp_name}', alpha=0.7)
        
        # Customize plots
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Training Time per Epoch')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.figures_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig
    
    def create_performance_comparison_table(self, results, save_path=None):
        """Create a comprehensive performance comparison table"""
        comparison_data = []
        
        for exp_name, result in results.items():
            if 'error' in result:
                continue
                
            model_info = result.get('model_info', {})
            
            row = {
                'Experiment': exp_name,
                'Model Type': model_info.get('model_type', 'Unknown'),
                'Architecture': model_info.get('architecture', 'Unknown'),
                'Pretrained': model_info.get('pretrained', False),
                'Total Parameters': model_info.get('total_parameters', 0),
                'Trainable Parameters': model_info.get('trainable_parameters', 0),
                'Final Train Acc': result.get('final_train_accuracy', 0),
                'Final Val Acc': result.get('final_val_accuracy', 0),
                'Best Val Acc': result.get('best_val_accuracy', 0),
                'Final F1': result.get('final_f1_score', 0),
                'Training Time (s)': result.get('total_training_time', 0),
                'Avg Epoch Time (s)': result.get('avg_epoch_time', 0),
                'Converged': result.get('convergence_info', {}).get('converged', False),
                'Convergence Epoch': result.get('convergence_info', {}).get('convergence_epoch', None)
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Format numbers
        df['Total Parameters'] = df['Total Parameters'].apply(lambda x: f"{x:,}")
        df['Trainable Parameters'] = df['Trainable Parameters'].apply(lambda x: f"{x:,}")
        df['Final Train Acc'] = df['Final Train Acc'].apply(lambda x: f"{x:.4f}")
        df['Final Val Acc'] = df['Final Val Acc'].apply(lambda x: f"{x:.4f}")
        df['Best Val Acc'] = df['Best Val Acc'].apply(lambda x: f"{x:.4f}")
        df['Final F1'] = df['Final F1'].apply(lambda x: f"{x:.4f}")
        df['Training Time (s)'] = df['Training Time (s)'].apply(lambda x: f"{x:.2f}")
        df['Avg Epoch Time (s)'] = df['Avg Epoch Time (s)'].apply(lambda x: f"{x:.4f}")
        
        if save_path:
            df.to_csv(save_path, index=False)
        else:
            df.to_csv(self.results_dir / 'performance_comparison.csv', index=False)
        
        return df
    
    def plot_performance_comparison(self, results, metrics=['best_val_accuracy', 'final_f1_score'], save_path=None):
        """Create bar plots comparing different metrics across experiments"""
        # Prepare data
        exp_names = []
        model_types = []
        metric_values = {metric: [] for metric in metrics}
        
        for exp_name, result in results.items():
            if 'error' in result:
                continue
                
            exp_names.append(exp_name)
            model_types.append(result.get('model_info', {}).get('model_type', 'Unknown'))
            
            for metric in metrics:
                metric_values[metric].append(result.get(metric, 0))
        
        # Create subplots
        fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 8))
        if len(metrics) == 1:
            axes = [axes]
        
        colors = ['#1f77b4' if mt == 'CNN' else '#ff7f0e' for mt in model_types]
        
        for i, metric in enumerate(metrics):
            bars = axes[i].bar(range(len(exp_names)), metric_values[metric], color=colors, alpha=0.7)
            axes[i].set_title(f'{metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Experiments')
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].set_xticks(range(len(exp_names)))
            axes[i].set_xticklabels(exp_names, rotation=45, ha='right')
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values[metric]):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Add legend
        cnn_patch = plt.Rectangle((0, 0), 1, 1, fc='#1f77b4', alpha=0.7, label='CNN')
        vit_patch = plt.Rectangle((0, 0), 1, 1, fc='#ff7f0e', alpha=0.7, label='ViT')
        fig.legend(handles=[cnn_patch, vit_patch], loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.figures_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig
    
    def plot_data_regime_analysis(self, results, save_path=None):
        """Analyze performance across different data regimes"""
        # Categorize experiments by data regime
        regime_categories = {
            'standard': [],
            'sparse': [],
            'imbalanced': [],
            'corrupted': []
        }
        
        for exp_name, result in results.items():
            if 'error' in result:
                continue
                
            if 'sparse' in exp_name:
                regime_categories['sparse'].append((exp_name, result))
            elif 'imbalanced' in exp_name:
                regime_categories['imbalanced'].append((exp_name, result))
            elif any(corruption in exp_name for corruption in ['noisy', 'corrupted', 'blur', 'gaussian']):
                regime_categories['corrupted'].append((exp_name, result))
            else:
                regime_categories['standard'].append((exp_name, result))
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Across Data Regimes', fontsize=16, fontweight='bold')
        
        for i, (regime, experiments) in enumerate(regime_categories.items()):
            if not experiments:
                continue
                
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            cnn_accs = []
            vit_accs = []
            labels = []
            
            for exp_name, result in experiments:
                model_type = result.get('model_info', {}).get('model_type', 'Unknown')
                acc = result.get('best_val_accuracy', 0)
                
                if model_type == 'CNN':
                    cnn_accs.append(acc)
                elif model_type == 'ViT':
                    vit_accs.append(acc)
                
                labels.append(exp_name.replace(f'{regime}_', '').replace('cnn_', '').replace('vit_', ''))
            
            # Plot comparison
            x = np.arange(len(labels))
            width = 0.35
            
            if cnn_accs:
                ax.bar(x - width/2, cnn_accs[:len(labels)], width, label='CNN', alpha=0.7)
            if vit_accs:
                ax.bar(x + width/2, vit_accs[:len(labels)], width, label='ViT', alpha=0.7)
            
            ax.set_title(f'{regime.title()} Data Regime')
            ax.set_xlabel('Experiment Variant')
            ax.set_ylabel('Best Validation Accuracy')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.figures_dir / 'data_regime_analysis.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig
    
    def plot_confusion_matrix(self, experiment_name, class_names=None, save_path=None):
        """Plot confusion matrix for a specific experiment"""
        detailed = self.load_detailed_results(experiment_name)
        if detailed is None or not detailed['confusion_matrices']:
            print(f"No confusion matrix data found for {experiment_name}")
            return None
        
        # Use the final confusion matrix
        cm = np.array(detailed['confusion_matrices'][-1])
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(cm.shape[0])]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {experiment_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.figures_dir / f'confusion_matrix_{experiment_name}.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        return plt.gcf()
    
    def create_summary_report(self, results, save_path=None):
        """Create a comprehensive summary report"""
        report_lines = []
        report_lines.append("# CNN vs Vision Transformer Experiment Results")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Overall statistics
        total_experiments = len([r for r in results.values() if 'error' not in r])
        failed_experiments = len([r for r in results.values() if 'error' in r])
        
        report_lines.append(f"## Summary Statistics")
        report_lines.append(f"- Total successful experiments: {total_experiments}")
        report_lines.append(f"- Failed experiments: {failed_experiments}")
        report_lines.append("")
        
        # Best performing models
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            best_acc = max(valid_results.items(), key=lambda x: x[1].get('best_val_accuracy', 0))
            best_f1 = max(valid_results.items(), key=lambda x: x[1].get('final_f1_score', 0))
            
            report_lines.append(f"## Best Performing Models")
            report_lines.append(f"- Highest Accuracy: {best_acc[0]} ({best_acc[1]['best_val_accuracy']:.4f})")
            report_lines.append(f"- Highest F1 Score: {best_f1[0]} ({best_f1[1]['final_f1_score']:.4f})")
            report_lines.append("")
        
        # Model comparison
        cnn_results = {k: v for k, v in valid_results.items() 
                      if v.get('model_info', {}).get('model_type') == 'CNN'}
        vit_results = {k: v for k, v in valid_results.items() 
                      if v.get('model_info', {}).get('model_type') == 'ViT'}
        
        if cnn_results and vit_results:
            cnn_avg_acc = np.mean([r['best_val_accuracy'] for r in cnn_results.values()])
            vit_avg_acc = np.mean([r['best_val_accuracy'] for r in vit_results.values()])
            
            report_lines.append(f"## Model Type Comparison")
            report_lines.append(f"- CNN Average Accuracy: {cnn_avg_acc:.4f}")
            report_lines.append(f"- ViT Average Accuracy: {vit_avg_acc:.4f}")
            report_lines.append(f"- Winner: {'CNN' if cnn_avg_acc > vit_avg_acc else 'ViT'}")
            report_lines.append("")
        
        # Detailed results
        report_lines.append(f"## Detailed Results")
        for exp_name, result in valid_results.items():
            if 'error' in result:
                continue
            
            model_info = result.get('model_info', {})
            report_lines.append(f"### {exp_name}")
            report_lines.append(f"- Model: {model_info.get('model_type', 'Unknown')} ({model_info.get('architecture', 'Unknown')})")
            report_lines.append(f"- Parameters: {model_info.get('total_parameters', 0):,}")
            report_lines.append(f"- Best Validation Accuracy: {result.get('best_val_accuracy', 0):.4f}")
            report_lines.append(f"- Final F1 Score: {result.get('final_f1_score', 0):.4f}")
            report_lines.append(f"- Training Time: {result.get('total_training_time', 0):.2f}s")
            report_lines.append(f"- Converged: {result.get('convergence_info', {}).get('converged', False)}")
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        else:
            with open(self.results_dir / 'experiment_summary_report.md', 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def generate_all_visualizations(self, experiment_pattern="*_summary.json"):
        """Generate all standard visualizations"""
        print("Loading experiment results...")
        results = self.load_experiment_results(experiment_pattern)
        
        if not results:
            print("No experiment results found!")
            return
        
        print(f"Found {len(results)} experiments")
        
        # Create performance comparison table
        print("Creating performance comparison table...")
        df = self.create_performance_comparison_table(results)
        print(f"Performance table saved with {len(df)} experiments")
        
        # Create performance comparison plots
        print("Creating performance comparison plots...")
        self.plot_performance_comparison(results)
        
        # Create training curves for a subset of experiments
        print("Creating training curves...")
        experiment_names = list(results.keys())[:6]  # Limit to avoid clutter
        self.plot_training_curves(experiment_names)
        
        # Create data regime analysis
        print("Creating data regime analysis...")
        self.plot_data_regime_analysis(results)
        
        # Create confusion matrices for key experiments
        print("Creating confusion matrices...")
        key_experiments = [name for name in experiment_names[:3]]  # Top 3
        for exp_name in key_experiments:
            self.plot_confusion_matrix(exp_name)
        
        # Create summary report
        print("Creating summary report...")
        self.create_summary_report(results)
        
        print(f"All visualizations saved to {self.figures_dir}")
        return results


if __name__ == '__main__':
    # Example usage
    visualizer = ExperimentVisualizer()
    results = visualizer.generate_all_visualizations()
    
    print("\nVisualization complete!")
    print(f"Check the {visualizer.figures_dir} directory for all generated plots.")
