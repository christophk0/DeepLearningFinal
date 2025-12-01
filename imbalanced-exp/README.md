# Imbalanced Dataset Experiments

This folder contains experiments comparing CNN (ResNet) and Vision Transformer (ViT) performance on imbalanced CIFAR10 datasets.

## Files

- `create_imbalanced_dataset.py`: Utilities to create imbalanced versions of CIFAR10
  - Long-tail imbalance: Exponential decay of samples across classes
  - Step imbalance: Some classes are minority, others are majority

- `run_imbalanced_experiment.py`: Main experiment script that:
  - Creates imbalanced training dataset
  - Trains both CNN and ViT models
  - Evaluates with comprehensive metrics (accuracy, F1-score, per-class recall/precision)
  - Generates visualizations and saves results

- `evaluate.py`: Evaluation utilities including:
  - Model evaluation using `AccumTensor` from `cka/metrics.py` for metric accumulation
  - Confusion matrix generation
  - Per-class metrics comparison plots

## Usage

### Basic Usage

Run with default settings (long-tail imbalance, ratio=0.1, 10 epochs):

```bash
cd imbalanced-exp
python run_imbalanced_experiment.py
```

### Custom Parameters

```bash
python run_imbalanced_experiment.py \
    --imbalance_type long_tail \
    --imbalance_ratio 0.1 \
    --num_epochs 10 \
    --config ../config.yaml \
    --output_dir ./results
```

### Options

- `--imbalance_type`: Type of imbalance (`long_tail` or `step`)
- `--imbalance_ratio`: Ratio between smallest and largest class (e.g., 0.1 means smallest has 10% of largest)
- `--num_epochs`: Number of training epochs
- `--config`: Path to config.yaml (default: ../config.yaml)
- `--output_dir`: Directory to save results (default: ./results)

## Output

Each experiment creates a timestamped directory in the output folder containing:

- `results.json`: Complete metrics and configuration
- `class_distribution.json`: Distribution of samples per class in training set
- `confusion_matrix_cnn.png`: Confusion matrix for CNN
- `confusion_matrix_vit.png`: Confusion matrix for ViT
- `per_class_metrics_comparison.png`: Side-by-side comparison of per-class metrics
- `training_curves.png`: Training loss curves for both models

## Metrics

The experiments evaluate:

- **Overall Accuracy**: Percentage of correct predictions
- **F1-Score (Macro)**: Unweighted mean of per-class F1-scores
- **F1-Score (Weighted)**: Weighted mean of per-class F1-scores (by support)
- **Per-Class Recall**: True positive rate for each class
- **Per-Class Precision**: Positive predictive value for each class
- **Per-Class F1-Score**: Harmonic mean of precision and recall for each class

## Example Experiments

### Long-tail imbalance (exponential decay)

```bash
python run_imbalanced_experiment.py --imbalance_type long_tail --imbalance_ratio 0.05 --num_epochs 15
```

### Step imbalance (half classes are minority)

```bash
python run_imbalanced_experiment.py --imbalance_type step --imbalance_ratio 0.1 --num_epochs 15
```

## Dependencies

Required packages (in addition to project dependencies):
- torchmetrics (for AccumTensor metric class, used in cka/metrics.py)
- matplotlib (for plotting)
- seaborn (for heatmaps)

Install with:
```bash
pip install torchmetrics matplotlib seaborn
```

Note: The evaluation uses `AccumTensor` from `cka/metrics.py` for metric accumulation, following the same pattern as CKA experiments.

## Google Colab Setup

To run experiments in Google Colab:

1. **Open the notebook:**
   - Upload `imbalanced_experiment_colab.ipynb` to Google Colab
   - Or see `COLAB_SETUP.md` for detailed instructions

2. **Enable GPU:**
   - Runtime → Change runtime type → Select GPU (T4 recommended)

3. **Upload required files:**
   - `CNN.py`, `VisionTransormer.py`, `config.yaml` (to root)
   - All files from `imbalanced-exp/` folder
   - `cka/metrics.py` (to `cka/` folder)

4. **Run the cells in order**

For detailed Colab setup instructions, see `COLAB_SETUP.md`.
