# Liver Tumor Segmentation using MONAI

A medical imaging project for automatic liver tumor segmentation using the MONAI framework. This repository contains preprocessing, training, and validation notebooks along with sample results.

## Features
- Uses MONAI for medical imaging deep learning workflows
- Reproducible preprocessing, training, and validation notebooks
- Validation metrics (Dice score) and example segmentation result image

## Repository Structure
- `Pre_proc.ipynb`: Data preprocessing pipeline (Getting only tumor masks, and segmenting liver crops)
- `training.ipynb`: Model training workflow
- `Validation_crosscheck.ipynb`: Validation and cross-check of predictions
- `dice_scores_val (1).csv`: Validation Dice scores
- `results.png`: Example prediction/segmentation result

## Requirements
- Python 3.9+
- Recommended: create a virtual environment
- Key packages: `monai`, `torch`, `numpy`, `pandas`, `matplotlib`, `scikit-image`, `nibabel` (for NIfTI)

You can install dependencies like:
```bash
pip install monai torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or cpu index
pip install numpy pandas matplotlib scikit-image nibabel tqdm jupyter
```


## Data
- This project expects medical imaging volumes (e.g., NIfTI `.nii`/`.nii.gz`) and corresponding labels.
- Update paths in the notebooks to point to your dataset locations.


## How to Run
1. Clone the repository:
```bash
git clone https://github.com/hjoshi95/Liver-Tumor-Segmentation-using-MONAI-.git
cd Liver-Tumor-Segmentation-using-MONAI-
```

## Results
- See `results.png` for the training curves and metrics.
- Validation Dice scores are stored in `dice_scores_val (1).csv`.

## Notes
- MONAI docs: https://docs.monai.io/
- PyTorch docs: https://pytorch.org/docs/
- Ensure GPU drivers/CUDA match your PyTorch install if using GPU.


