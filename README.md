# RSNA Pneumonia Detection: Modular PyTorch Pipeline (Phase 1 MVP)

## 📌 Executive Summary
This repository contains a modularized, end-to-end PyTorch training pipeline for pneumonia detection using the RSNA clinical dataset. Designed with MLOps principles, the project decouples data acquisition, model definition, and training logic, migrating away from monolithic Jupyter Notebooks into scalable, environment-agnostic Python scripts.

## 🏗️ Architecture & Project Structure
The pipeline is strictly modularized to ensure reproducibility and ease of deployment on remote Linux/Colab instances:
```
├── notebooks/
│   └── colab_runner.ipynb
├── scripts/
│   └── setup_data.sh      
├── dataset.py              
├── model.py                
├── train.py                
├── main.py                
├── utils.py
└── README.md
```
## 🚀 Quick Start (Reproducibility)
To execute the pipeline in a cloud environment (e.g., Google Colab), simply use the provided runner:
1. Open `notebooks/colab_runner.ipynb` in Colab. 
2. The runner will automatically clone this repository, execute `setup_data.sh` to fetch the 4GB dataset, and trigger `main.py`.

## ⚠️ Known Issues & Phase 2 Roadmap
**[Critical] Severe Overfitting on Validation Set:**
In the current MVP stage (Phase 1), the model achieves near-zero training loss while validation loss spikes significantly. This is a recognized architectural behavior stemming from the following MVP trade-offs and dataset characteristics:

1. **Absence of Data Augmentation:** To validate the script-based E2E pipeline rapidly, `Albumentations` (rotation, cropping, contrast adjustments) were temporarily bypassed. Consequently, the over-parameterized ResNet-18 model memorizes pixel layouts rather than learning generalizable lung opacities.
2. **Unaddressed Class Imbalance:** The RSNA dataset features a heavily skewed distribution towards non-pneumonia classes. The omission of `class_weights` in the `CrossEntropyLoss` function during Stage 1 accelerates majority-class memorization.
3. **Confounding Artifacts (Shortcut Learning):** Unlike curated pediatric pneumonia datasets, the RSNA clinical dataset contains numerous clinical artifacts (e.g., chest tubes, portable X-ray markers). Without aggressive regularization and transformations, ResNet-18 latches onto these confounding variables as shortcuts for classification.
4. **Unconstrained Backbone Fine-Tuning:** Updating all weights of the pre-trained ResNet-18 backbone simultaneously allowed early convolutional layers to overwrite robust, generic feature extractors (e.g., ImageNet-learned edges and textures) with task-specific clinical noise.

**Next Immediate Actions (Phase 2):**
- [x] Implement a robust `setup_data.sh` to fully automate Kaggle API downloads and decouple data preparation from the Python runtime.
- [x] Include a `seed_everything` function in `main.py` to ensure reproducibility.
- [ ] Add argparse for setting parameters from the outside.
- [ ] Re-integrate data augmentations (via `Albumentations`) into `main.py` to disrupt pixel-level memorization.
- [x] Refactor `dataset.py` to eliminate Pandas `.loc` bottlenecks in the `__getitem__` method for optimal GPU utilization.
- [ ] Introduce dynamic `class_weights` and Early Stopping mechanisms to enforce proper generalization.
- [ ] Implement layer-freezing mechanics (e.g., freeze early ResNet blocks) to leverage stable transfer learning features and mitigate representation drift.