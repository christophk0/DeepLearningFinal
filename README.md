# CNN vs Vision Transformer Comparison

A comprehensive experimental comparison of Convolutional Neural Networks and Vision Transformers across different data regimes and architectural variations.

## Project Structure

```
DeepLearningFinal/
├── CNN.py                    # CNN model implementation (ResNet variants)
├── VisionTransormer.py       # Vision Transformer implementation
├── local_attention.py        # Local attention mechanism for ViT
├── experiments.py            # Main experiment orchestration
├── metrics.py               # Comprehensive metrics tracking
├── utils.py                 # Data loading and utility functions
├── data_regimes.py          # Data regime generators (sparse, imbalanced, corrupted)
├── visualization.py         # Results visualization and analysis
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── results_summary.md       # Comprehensive results analysis
├── results/                 # Experiment outputs
│   ├── logs/               # Training logs
│   ├── checkpoints/        # Model checkpoints
│   ├── summaries/          # Experiment summaries
│   └── figures/            # Generated visualizations
└── data/                   # Dataset storage
```

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd DeepLearningFinal

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Experiments

```bash
# Run all experiments
python experiments.py --experiment_set all

# Run specific experiment sets
python experiments.py --experiment_set baseline
python experiments.py --experiment_set data_regime
python experiments.py --experiment_set architectural
```

### 3. Generate Visualizations

```bash
# Generate all visualizations and analysis
python visualization.py
```

## Experiment Sets

### Baseline Experiments
Compares pretrained vs from-scratch training for both CNN and ViT on standard CIFAR-10:
- `cnn_pretrained_cifar10`
- `cnn_scratch_cifar10`
- `vit_pretrained_cifar10`
- `vit_scratch_cifar10`

### Data Regime Experiments
Tests model performance across different data conditions:
- **Sparse data**: 1%, 5%, 10%, 25%, 50% of training data
- **Imbalanced data**: Skewed class distributions
- **Corrupted data**: Gaussian noise, blur, and other corruptions

### Architectural Experiments
Explores different architectural variations:
- **Shallow networks**: Reduced depth CNN and ViT
- **Local attention**: ViT with restricted attention windows
- **Layer dropping**: Analysis of different network depths

## Key Features

### Model Implementations
- **CNN**: ResNet-18/34/50 with configurable depth and pretraining
- **ViT**: Vision Transformer B/16, B/32, L/16 variants
- **Local ViT**: Custom implementation with local attention mechanism

### Data Regime Support
- **Sparse datasets**: Balanced and random sampling
- **Imbalanced datasets**: Step and exponential imbalance patterns
- **Corrupted datasets**: 8 corruption types with 5 severity levels

### Comprehensive Metrics
- Accuracy, F1-score, per-class precision/recall
- Training time, convergence detection
- Parameter counting, FLOP estimation
- Confusion matrices, attention visualizations

### Visualization Suite
- Training curves comparison
- Performance comparison tables and plots
- Data regime analysis
- Confusion matrices
- Summary reports

## Configuration

The `config.yaml` file contains all experimental settings:

```yaml
# General settings
batch_size: 64
num_epochs: 10
model_type: 'vt'  # 'cnn' or 'vt'

# CNN configuration
cnn:
  architecture: 'resnet18'
  pretrained: true
  learning_rate: 0.001
  # ... more settings

# ViT configuration
vision_transformer:
  architecture: 'vit_b_16'
  pretrained: true
  use_local_attention: false
  local_window_size: 7
  # ... more settings
```

## Results and Analysis

### Performance Comparison
All results are automatically saved to `results/` directory:
- **Summaries**: JSON files with detailed metrics
- **Logs**: Training progress and hyperparameters
- **Checkpoints**: Best model weights
- **Figures**: Generated visualizations

### Key Findings
See `results_summary.md` for comprehensive analysis including:
- Performance across different data regimes
- Architectural comparison insights
- Training dynamics analysis
- Practical recommendations

## Research Questions Addressed

1. **Data Efficiency**: Under what data conditions do ViTs outperform CNNs?
2. **Architectural Properties**: What explains performance differences?
3. **Pretraining Impact**: How does pretraining affect the comparison?
4. **Local Attention**: What are the benefits of local attention in ViTs?

## Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended (MPS support for Apple Silicon)
- **Memory**: 8GB+ RAM, 4GB+ GPU memory
- **Storage**: 10GB+ for datasets and results

## Reproducibility

- Fixed random seeds across all experiments
- Deterministic training procedures
- Comprehensive logging of hyperparameters
- Version-controlled dependencies

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with appropriate tests
4. Submit a pull request

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{cnn_vit_comparison_2024,
  title={Comprehensive Comparison of CNNs and Vision Transformers Across Data Regimes},
  author={[Team QCTM]},
  year={2024},
  howpublished={\url{<repository-url>}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please contact:
- Christoph Kinzel: ckinzel6@gatech.edu
- Joyce Gu: joyce_gu@gatech.edu
- Tomás Valdivia Hennig: thennig3@gatech.edu
- Mark Gardner: mgardner60@gatech.edu
