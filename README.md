# Deep Learning for Skin Lesion Classification

This repository contains the implementation and experimentation code for the thesis project: **"Deep Learning for Skin Lesion Classification"** by *Thanetpol Noynuai*, submitted in partial fulfillment of the Bachelor of Engineering degree at King Mongkut’s Institute of Technology Ladkrabang (KMITL), 2024.

## Project Objective

The primary goal of this project is to develop and evaluate various deep learning architectures to classify skin lesions accurately, aiding in the early diagnosis of skin cancer, particularly **melanoma** and **basal cell carcinoma**. This work leverages the **HAM10000** dataset and investigates solutions to class imbalance, overfitting, and hardware memory limitations.

## Project Structure

This project is organized by architecture and experimental setup. Below is an overview:

### InceptionV3/
- Contains experiments with the InceptionV3 model.
- Subfolders:
  - `test shuffle/`: Experiments with different shuffle settings.
  - `test resample/`: Experiments involving class balancing via resampling.
- Includes baseline and variant training notebooks.

### InceptionResNetV2/
- Experiments using InceptionResNetV2.
- Subfolders:
  - `test shuffle/`: Shuffle settings.
  - test resample/without resample and `with resample`: To compare effect of resampling.
  - `test dropout/`: Models with/without dropout layers.

### DenseNet121/
- Notebooks for testing DenseNet121 performance and regularization.

### `Xception/`, `VGG16/`, `VGG19/`, `ResNet50/`, `ResNet101/`, NASNetMobile/
- Each contains experiments using the respective pretrained models.
- All follow the without resample and regularizationX.ipynb naming pattern.

### SimpleCnn/
- Contains a self-built CNN architecture.
- Includes versions with different training configurations (checkpoint, 5-fold, class-wise).

### `5 fold.ipynb`, 5 fold InceptionResNetV2.1.ipynb
- 5-fold cross-validation implementations to assess model generalization.

## Methodology

- **Dataset**: [HAM10000](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)
- **Preprocessing**: Resizing, normalization, and augmentation using `ImageDataGenerator`.
- **Model Architectures**:
  - SimpleCNN (custom architecture)
  - Pretrained: VGG16/19, InceptionV3, InceptionResNetV2, Xception, ResNet50/101, DenseNet121, NASNetMobile
- **Techniques Applied**:
  - Dropout, L2 Regularization
  - Early Stopping, ReduceLROnPlateau
  - Custom dense layer stack to prevent OOM issues
  - 5-fold cross-validation
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, AUC

## Results Summary

Top 3 performing models (based on validation and test metrics):
1. **InceptionResNetV2**
2. **InceptionV3**
3. **Xception**

Below are the performance summaries and confusion matrices for each model:

### 📊 InceptionResNetV2 Results
![InceptionResNetV2 Results](Result%20of%20InceptionResNetV2.png)

### 📊 InceptionV3 Results
![InceptionV3 Results](Result%20of%20InceptionV3.png)

### 📊 Xception Results
![Xception Results](Result%20of%20Xception.png)


Refer to the thesis PDF and corresponding Jupyter notebooks for full evaluation metrics and confusion matrices.

## RequirementsAdd commentMore actions

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy, Matplotlib, Pandas
- Jupyter Notebook
- (Optional) GPU environment for training (Recommended: >=8GB VRAM)

Install dependencies:
```bash
pip install -r requirements.txt
