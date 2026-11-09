# Human Activity Recognition Using Deep Learning and Edge AI

Human Activity Recognition using TensorFlow/Keras and optimized TensorFlow Lite models for edge inference.

## Overview

This project classifies six human activities from smartphone sensor features:

- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

The project covers dataset exploration, preprocessing, model training, evaluation, TensorFlow Lite conversion, quantization, and edge-inference benchmarking.

## Dataset

This project uses the [UCI Human Activity Recognition Using Smartphones Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).

Each sample contains 561 engineered time- and frequency-domain features. The original dataset separates training and testing subjects, which helps measure generalization to people not seen during training.

The dataset is not included in this repository. Download and extract it from the official UCI page before running the notebook.

## Repository Contents

```text
human-activity-recognition-edge-ai/
├── README.md
└── UCI HAR Dataset/
    └── UCI HAR Dataset/
        ├── model.ipynb
        ├── requirements.txt
        ├── train/
        ├── test/
        ├── features.txt
        ├── features_info.txt
        └── activity_labels.txt
```

The trained model artifacts are also included:

- `har_baseline.keras`
- `har_baseline.tflite`
- `har_quantized.tflite`

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r "UCI HAR Dataset\UCI HAR Dataset\requirements.txt"
```

Open `UCI HAR Dataset\UCI HAR Dataset\model.ipynb` in Jupyter or VS Code and run the cells in order.

## Training

### Data loading

The notebook loads:

- `train/X_train.txt` as the 561-feature input matrix
- `train/y_train.txt` as the activity labels
- `train/subject_train.txt` as the subject identifiers
- The corresponding files from `test/` for final evaluation

The feature matrix has 7,352 training samples and 561 features. The test set has 2,947 samples and 561 features.

The dataset features are already normalized by the dataset creators. Labels are converted from `1–6` to zero-based `0–5` labels for Keras sparse categorical classification.

### Validation strategy

The official test set is kept untouched until final evaluation. A validation set is created from complete subjects in the original training set rather than randomly splitting individual rows.

This prevents samples from the same person appearing in both the training and validation sets, giving a more realistic estimate of performance on unseen subjects.

### Baseline architecture

The baseline is a fully connected neural network:

```text
Input: 561 features
Dense: 128 units, ReLU
Dropout: 0.3
Dense: 64 units, ReLU
Dropout: 0.3
Output: 6 units, Softmax
```

### Training configuration

The model is compiled with:

- **Optimizer:** Adam
- **Loss:** sparse categorical cross-entropy
- **Metric:** accuracy
- **Batch size:** 64
- **Maximum epochs:** 50
- **Random seed:** 42

Adam updates the model weights using gradients. Sparse categorical cross-entropy is appropriate because each sample belongs to one of six classes and the labels are stored as integers.

Dropout is used as regularization to reduce overfitting. Early stopping monitors validation loss with a patience of eight epochs and restores the weights from the best validation epoch.

## Evaluation

The trained model is evaluated on the untouched test set using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The baseline achieved a test accuracy of approximately **93.52%**. The notebook contains the complete classification report and confusion matrix.

## TensorFlow Lite and Optimization

The trained Keras model is converted to TensorFlow Lite for lightweight inference. The float32 TFLite model accepts an input with shape `(1, 561)` and produces six output scores.

Dynamic-range quantization is then applied to reduce the size of the model weights while keeping the input interface as `float32`.

| Model | Size | Accuracy | Average latency |
|---|---:|---:|---:|
| Keras | 0.9508 MB | 93.52% | See notebook |
| TFLite float32 | 0.3095 MB | 93.52% | 0.0112 ms |
| TFLite dynamic-range | 0.0829 MB | 93.52% | 0.0093 ms |

Latency depends on the hardware and runtime environment, so these values should be treated as measurements from the development machine rather than universal guarantees.

## Model Artifacts

- `har_baseline.keras` — original trained Keras model
- `har_baseline.tflite` — float32 TensorFlow Lite model
- `har_quantized.tflite` — dynamic-range quantized TensorFlow Lite model

## Limitations and Future Work

- The model uses pre-engineered features rather than raw sensor sequences.
- Performance may change for different phones, sensor placements, or users.
- Latency should be benchmarked on the target mobile or embedded device.
- Future work could compare Adam with SGD or AdamW, test full integer quantization, and create a real-time sensor demonstration.

## Citation

Please acknowledge the original publication:

> Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. ESANN 2013.
