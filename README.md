# Cow Teat Image Classification using CNN with PyTorch

A custom Convolutional Neural Network (CNN) for classifying cow teat images into 4 categories. This project was developed as part of a course assignment and compares against the VGG16 baseline from the [SCTL paper](https://www.mdpi.com/2076-2615/12/7/886/htm).

## Overview

- **Task**: Multi-class image classification on cow teat dataset
- **Model**: Custom 5-layer CNN with ImageNet normalization
- **Framework**: PyTorch
- **Dataset**: Cow teat images (4 classes)
- **Accuracy**: ~61.3% (Base: ~61%, VGG16 Target: >66.8%)

## Project Structure

```
Neural-project/
├── README.md
├── requirements.txt
├── cow-teat-classification.ipynb   # Main training & evaluation notebook
├── data/                           # Place your dataset here
│   ├── Train/                      # ImageFolder format (class subfolders)
│   │   ├── class_0/
│   │   ├── class_1/
│   │   ├── class_2/
│   │   └── class_3/
│   └── Test/                       # Images with labels in filename (e.g. c1_image.jpg)
├── outputs/
│   ├── model.pt                    # Trained weights
│   └── predictions.csv
└── Image classification using CNN with pytorch.pdf
```

## Dataset Setup

1. **Training data**: Organize images in `data/Train/` with subfolders per class:
   ```
   Train/
   ├── class_0/
   ├── class_1/
   ├── class_2/
   └── class_3/
   ```

2. **Test data**: Place images in `data/Test/`. Filenames should encode the label (e.g. `c1_xxx.jpg` where the digit after `c` is the class index).

## Installation

```bash
# Clone the repository
git clone https://github.com/bhansalijainam/Neural-project.git
cd Neural-project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

For GPU training, use [Google Colab](https://colab.research.google.com/) or [Kaggle](https://www.kaggle.com/) with GPU enabled.

## Usage

### Run without the dataset (demo mode)

The notebook can run **end-to-end without the cow teat dataset** for quick showcase (e.g. on Colab or after cloning):

1. Open `cow-teat-classification.ipynb`.
2. In the **Config** cell, keep `USE_DEMO_DATA = True` (default).
3. Run all cells. The notebook uses synthetic data, trains for 3 epochs, and writes outputs to `outputs/`.

### Run with the real dataset

1. **Set data paths**:
   - The project expects data in `Train_1/`, `Train_2/`, and `Test-2/` (as configured in the script).
   - If using the original folder structure, ensure paths in `run_training.py` or the notebook match.

2. **Run the training**:
   - **Recommended**: Run the Python script for best performance:
     ```bash
     python run_training.py
     ```
   - **Alternative**: Open `cow-teat-classification.ipynb` in Jupyter/Colab.

3. **Outputs**:
   - Trained model: `outputs/model.pt`
   - Predictions: `outputs/test_results.txt`
   - Training loss plot: `outputs/training_loss.png`

## Model Architecture

| Layer | Type | Output |
|-------|------|--------|
| 1 | Conv2d(3→32, 3×3) + ReLU + MaxPool | 112×112 |
| 2 | Conv2d(32→64, 3×3) + ReLU + MaxPool | 56×56 |
| 3 | Conv2d(64→128, 3×3) + ReLU + MaxPool | 28×28 |
| 4 | Conv2d(128→256, 3×3) + ReLU + MaxPool | 14×14 |
| 5 | Conv2d(256→512, 3×3) + ReLU + MaxPool | 7×7 |
| 6 | FC(512×7×7 → 1024) + ReLU | |
| 7 | FC(1024 → 512) + ReLU | |
| 8 | FC(512 → 4) | 4 classes |

## Pre-trained Weights

Model weights (~20MB) are hosted on Google Drive:
[Download model.pt](https://drive.google.com/file/d/1vFQYmbQCun5k5K5LNLWxjNQuKT-i2ObU/view?usp=sharing)

## Paper

📄 [Image Classification using CNN with PyTorch (PDF)](./Image%20classification%20using%20CNN%20with%20pytorch.pdf)

## References

- [SCTL Paper](https://www.mdpi.com/2076-2615/12/7/886/htm) — VGG16 baseline: 66.8%

## Troubleshooting: Notebook won't open / "Could not register service worker"

If Cursor shows **"Could not initialize webview"** or **"Could not register service worker"** when opening the `.ipynb` file:

1. **Quit Cursor completely** (Cmd+Q on Mac; ensure no Cursor process is running).
2. **Clear Cursor's cache** (macOS):
   ```bash
   rm -rf ~/Library/Application\ Support/Cursor/Cache
   rm -rf ~/Library/Application\ Support/Cursor/Service\ Worker 2>/dev/null || true
   ```
3. **Reopen Cursor** and open the project again; try opening the notebook.

**If it still fails**, run the notebook outside Cursor:

- **Jupyter in browser:** From the project folder run `jupyter notebook` or `jupyter lab`, then open `cow-teat-classification.ipynb`.
- **VS Code:** Open the same folder in VS Code and open the notebook there (after clearing VS Code cache if needed).

## License

Please refer to the course guidelines for usage and attribution.
