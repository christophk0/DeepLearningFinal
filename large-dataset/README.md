# Large Dataset Experiments

This folder contains experiments comparing CNN (ResNet) and Vision Transformer (ViT) performance on large datasets.

## Overview

The experiments train both CNN and ViT models on large-scale datasets (currently CIFAR-100) and compare their performance in terms of:
- Accuracy
- F1-Score (Macro and Weighted)
- Per-class metrics (Recall, Precision, F1-Score)
- Training and validation loss curves

## Files

- `large_dataset_experiment_colab.ipynb`: Jupyter notebook for running experiments in Google Colab
- `run_large_dataset_experiment.py`: Main Python script to run experiments
- `evaluate.py`: Evaluation metrics and visualization functions
- `requirements.txt`: Python package dependencies

## Usage

### Running locally

**CIFAR-100:**
```bash
cd large-dataset
python run_large_dataset_experiment.py \
    --dataset cifar100 \
    --num_epochs 10 \
    --config ../config.yaml \
    --output_dir ./results
```

**Coyo-labeled-300m:**
```bash
cd large-dataset
python run_large_dataset_experiment.py \
    --dataset coyo \
    --num_epochs 10 \
    --max_samples 50000 \
    --num_classes 500 \
    --config ../config.yaml \
    --output_dir ./results
```

Note: For Coyo-labeled-300m, you can adjust:
- `--max_samples`: Maximum number of samples to load (default: 100000)
- `--num_classes`: Number of top classes to use (default: 1000)

### Running in Google Colab

1. Upload the notebook `large_dataset_experiment_colab.ipynb` to Google Colab
2. Upload required files:
   - `CNN.py` and `VisionTransormer.py` from parent directory
   - `config.yaml` from parent directory
   - All files from `large-dataset/` folder
   - `cka/metrics.py` and `cka/hook_manager.py` (if needed)
3. Run the notebook cells sequentially

## Supported Datasets

- **CIFAR-100**: 100 classes, 50,000 training images, 10,000 test images
- **Coyo-labeled-300m**: Large-scale dataset with 300M images, multi-label annotations. 
  - Default: Top 1000 classes, 100k samples
  - Configurable via `--max_samples` and `--num_classes` arguments
  - Images are downloaded on-demand from URLs

## Output

Results are saved in `results/large_dataset_<dataset>_<timestamp>/` with:
- `results.json`: All metrics in JSON format
- `charts/`: Folder containing all visualization charts:
  - `confusion_matrix_cnn.png`: Confusion matrix for CNN
  - `confusion_matrix_vit.png`: Confusion matrix for ViT
  - `per_class_metrics_comparison.png`: Comparison of per-class metrics
  - `training_curves.png`: Training and validation loss curves

## Notes

- For CIFAR-100 (100 classes), visualizations are automatically adjusted for large number of classes
- The models automatically adapt their final classification layer to match the number of classes in the dataset
- Both models use pretrained weights by default (can be configured in `config.yaml`)
