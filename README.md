

````markdown
# 🧠 Deep Learning for Medical Image Classification
### CNN-Based Medical Image Classification Pipeline with TensorFlow & Keras

<p align="center">

  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/Medical%20AI-Research-4CAF50?style=for-the-badge" />

</p>

<p align="center">
  <strong>End-to-end CNN pipeline for automated medical image classification.</strong>
</p>

---

# 📌 Project Overview

This project implements an end-to-end **deep learning pipeline for medical image classification** using a custom Convolutional Neural Network (CNN) built with **TensorFlow and Keras**.

The system is designed to transform raw medical images into statistically validated predictions through a complete machine learning workflow:

```text
Medical Images
      │
      ▼
┌───────────────────────┐
│ Dataset Ingestion     │
│ & Validation          │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Image Preprocessing   │
│ Resize / Normalize    │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Data Augmentation     │
│ Rotation / Flip /     │
│ Shear / Transformation│
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ CNN Feature Extraction│
│ Conv → BN → ReLU      │
│ → Pooling             │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Classification Head   │
│ Dense + Dropout       │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Model Evaluation      │
│ Precision / Recall /  │
│ F1 / Confusion Matrix │
└───────────────────────┘
````

> **Important:** This project is intended for research and educational purposes. Predictions from the model should not be interpreted as a clinical diagnosis.

---

# 🎯 Objectives

The primary objectives of this project are:

* Build a CNN from scratch using TensorFlow/Keras.
* Create a reproducible medical image preprocessing pipeline.
* Reduce overfitting through augmentation and regularization.
* Preserve class balance during dataset partitioning.
* Evaluate performance using clinically meaningful metrics.
* Visualize training behavior and classification errors.
* Establish a foundation for future transfer-learning experiments.

---

# 🏗️ System Architecture

```mermaid
flowchart TD

    A[Medical Image Dataset] --> B[Data Validation]

    B --> C[Train / Validation / Test Split]

    C --> D[Image Preprocessing]

    D --> E[Normalization]

    E --> F[Data Augmentation]

    F --> G[CNN Feature Extraction]

    G --> H[Batch Normalization]

    H --> I[Activation Function]

    I --> J[Max Pooling]

    J --> K[Dropout]

    K --> L[Flatten]

    L --> M[Dense Layers]

    M --> N[Final Classification Head]

    N --> O[Predictions]

    O --> P[Evaluation]

    P --> Q[Confusion Matrix]

    P --> R[Precision]

    P --> S[Recall]

    P --> T[F1 Score]
```

---

# 🧬 Machine Learning Pipeline

```mermaid
flowchart LR

    A["Raw Images"] --> B["Cleaning"]
    B --> C["Resize"]
    C --> D["Normalize"]
    D --> E["Augment"]

    E --> F["Training"]
    F --> G["Validation"]

    G --> H["Model Selection"]

    H --> I["Test Set"]

    I --> J["Performance Analysis"]

    J --> K["Classification Report"]
    J --> L["Confusion Matrix"]
    J --> M["Error Analysis"]
```

---

# 📂 Dataset Architecture

The dataset is dynamically loaded from a local directory or Google Drive environment.

### Dataset characteristics

| Property       | Description                                           |
| -------------- | ----------------------------------------------------- |
| Image format   | `.jpg`, `.jpeg`, `.png`                               |
| Task           | Binary / Multi-class classification                   |
| Input          | Medical images                                        |
| Training       | Model optimization                                    |
| Validation     | Hyperparameter and generalization monitoring          |
| Testing        | Final unbiased evaluation                             |
| Split strategy | Stratified partitioning                               |
| Preprocessing  | Resize + normalization                                |
| Augmentation   | Rotation, shear, flipping and related transformations |

Example structure:

```text
dataset/
│
├── train/
│   ├── class_0/
│   └── class_1/
│
├── validation/
│   ├── class_0/
│   └── class_1/
│
└── test/
    ├── class_0/
    └── class_1/
```

---

# 🔬 Data Processing Pipeline

```mermaid
flowchart TD

    A[Input Image] --> B[Load Image]

    B --> C[Resize to Target Dimensions]

    C --> D[Convert Pixel Values]

    D --> E[Normalize Pixel Intensity]

    E --> F{Training Dataset?}

    F -->|Yes| G[Data Augmentation]
    F -->|No| H[No Augmentation]

    G --> I[Model]
    H --> I
```

### Preprocessing objectives

The preprocessing stage standardizes the input space so that the CNN receives consistent tensors regardless of the original image dimensions.

Typical operations include:

```python
Image
  ↓
Resize
  ↓
Normalization
  ↓
Tensor Conversion
  ↓
CNN Input
```

---

# 🧠 CNN Architecture

The model follows a progressively hierarchical feature-learning strategy.

```text
INPUT
│
├── Conv2D
├── BatchNormalization
├── ReLU
├── MaxPooling2D
│
├── Conv2D
├── BatchNormalization
├── ReLU
├── MaxPooling2D
│
├── Dropout
│
├── Flatten
│
├── Dense
├── BatchNormalization
├── ReLU
│
├── Dropout
│
└── Output Layer
```

### Conceptual architecture

```mermaid
flowchart LR

    A["Input Image"] -->

    B["Conv2D"]

    B --> C["BatchNorm"]

    C --> D["ReLU"]

    D --> E["MaxPool"]

    E --> F["Conv2D"]

    F --> G["BatchNorm"]

    G --> H["ReLU"]

    H --> I["MaxPool"]

    I --> J["Dropout"]

    J --> K["Flatten"]

    K --> L["Dense"]

    L --> M["BatchNorm"]

    M --> N["ReLU"]

    N --> O["Dropout"]

    O --> P["Output"]
```

---

# 🧩 Layer Responsibilities

| Layer                | Purpose                                      |
| -------------------- | -------------------------------------------- |
| `Conv2D`             | Learns spatial features and local structures |
| `BatchNormalization` | Stabilizes intermediate activations          |
| `ReLU`               | Introduces non-linearity                     |
| `MaxPooling2D`       | Reduces spatial dimensions                   |
| `Dropout`            | Helps reduce overfitting                     |
| `Flatten`            | Converts feature maps into vectors           |
| `Dense`              | Performs high-level feature combination      |
| `Output Layer`       | Produces final classification probabilities  |

---

# ⚙️ Training Strategy

The training pipeline incorporates multiple mechanisms to improve generalization.

```mermaid
flowchart TD

    A["Training Batch"] --> B["Forward Pass"]

    B --> C["Prediction"]

    C --> D["Loss Calculation"]

    D --> E["Backpropagation"]

    E --> F["Gradient Update"]

    F --> G["Validation"]

    G --> H{"Validation Loss Improving?"}

    H -->|Yes| I["Continue Training"]

    H -->|No| J["Reduce Learning Rate"]

    J --> K["Continue Optimization"]

    K --> G
```

### Adaptive learning rate

`ReduceLROnPlateau` dynamically decreases the learning rate when validation performance stops improving.

This allows the model to:

* make larger updates during early training,
* perform finer optimization later,
* potentially escape training plateaus,
* improve convergence stability.

---

# 🛡️ Overfitting Control

Medical imaging datasets can be relatively small compared with general computer vision datasets.

This project therefore uses several regularization mechanisms:

```text
              Overfitting Risk
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
 Data Augmentation  Dropout   BatchNormalization
       │            │            │
       └────────────┼────────────┘
                    ▼
             Better Generalization
```

### Main techniques

**Data augmentation**

Creates additional training variation through transformations such as:

* rotation
* shear
* horizontal flipping
* vertical flipping
* spatial transformations

**Dropout**

Randomly disables a subset of neural connections during training.

**Batch normalization**

Normalizes intermediate activations and can make optimization more stable.

---

# 📊 Evaluation Framework

Accuracy alone is not sufficient for evaluating medical classification systems.

The evaluation pipeline therefore examines several complementary metrics.

```mermaid
flowchart TD

    A["Model Predictions"] --> B["Confusion Matrix"]

    B --> C["True Positive"]
    B --> D["True Negative"]
    B --> E["False Positive"]
    B --> F["False Negative"]

    C --> G["Precision"]
    D --> G

    E --> G

    C --> H["Recall"]

    F --> H

    G --> I["F1 Score"]
    H --> I
```

---

# 📐 Evaluation Mathematics

## Precision

Measures how many predicted positive cases were actually positive.

$$
Precision = \frac{TP}{TP + FP}
$$

---

## Recall / Sensitivity

Measures how many actual positive cases were successfully detected.

$$
Recall = \frac{TP}{TP + FN}
$$

---

## F1 Score

Balances precision and recall using their harmonic mean.

$$
F1 =
2
\times
\frac{Precision \times Recall}
{Precision + Recall}
$$

---

## Confusion Matrix

For binary classification:

```text
                    Actual
                ┌───────────────┐
                │ Positive │ Negative │
───────────────┼──────────┼───────────┤
Predicted      │          │          │
Positive       │    TP    │    FP    │
───────────────┼──────────┼───────────┤
Negative       │    FN    │    TN    │
───────────────┴──────────┴───────────┘
```

This allows detailed inspection of which classes the model confuses.

---

# 📈 Training Visualization

One of the most important diagnostics is the relationship between training and validation performance.

Recommended charts:

```text
Accuracy

1.0 ┤                 ╭────────────
    │             ╭───╯
0.8 ┤         ╭───╯
    │     ╭───╯
0.6 ┤ ╭───╯
    │╭╯
0.4 ┼──────────────────────────────
    └──────────────────────────────
       Epoch →


Loss

1.2 ┤╲
    │ ╲
0.8 ┤  ╲
    │   ╲
0.4 ┤    ╲____
    │
0.0 ┼──────────────────────────────
    └──────────────────────────────
       Epoch →
```

The actual experiment should generate:

* Training Accuracy vs Validation Accuracy
* Training Loss vs Validation Loss
* Learning-rate progression
* Confusion matrix
* Per-class precision/recall/F1

---

# 🖼️ Recommended Model Diagnostics

The project can be extended with a complete visual analysis dashboard:

```mermaid
flowchart LR

    A["Training History"] --> B["Accuracy Curve"]

    A --> C["Loss Curve"]

    A --> D["Learning Rate"]

    E["Test Predictions"] --> F["Confusion Matrix"]

    E --> G["Classification Report"]

    E --> H["Misclassified Images"]

    E --> I["Confidence Distribution"]
```

### Suggested visualizations

| Visualization          | Purpose                          |
| ---------------------- | -------------------------------- |
| Accuracy curve         | Detect convergence               |
| Loss curve             | Detect overfitting               |
| Confusion matrix       | Understand class-specific errors |
| Precision/Recall chart | Compare class performance        |
| F1 score chart         | Compare balanced performance     |
| Prediction confidence  | Understand model certainty       |
| Misclassified samples  | Perform visual error analysis    |
| Learning-rate curve    | Analyze optimization behavior    |

---

# 🔍 Error Analysis

A strong medical AI pipeline should not only ask:

> "How accurate is the model?"

It should also ask:

> "Where does the model fail?"

The error-analysis workflow can be represented as:

```mermaid
flowchart TD

    A["Test Predictions"]

    A --> B{"Prediction Correct?"}

    B -->|Yes| C["Correct Samples"]

    B -->|No| D["Misclassified Samples"]

    D --> E["False Positive"]

    D --> F["False Negative"]

    E --> G["Visual Inspection"]

    F --> G

    G --> H["Identify Failure Patterns"]

    H --> I["Improve Dataset / Model"]
```

Potential sources of error include:

* insufficient training examples,
* class imbalance,
* noisy images,
* ambiguous pathology,
* inconsistent acquisition conditions,
* preprocessing artifacts,
* model overfitting.

---

# 📊 Suggested Performance Dashboard

A future experiment can present results using a dashboard such as:

```text
┌────────────────────────────────────────────────────┐
│                MODEL PERFORMANCE                   │
├───────────────┬───────────────┬────────────────────┤
│ Accuracy      │ Precision     │ Recall             │
│     XX.X%     │     XX.X%     │     XX.X%          │
├───────────────┼───────────────┼────────────────────┤
│ F1 Score      │ Test Samples  │ Classes            │
│     XX.X%     │     XXXX      │       X            │
└───────────────┴───────────────┴────────────────────┘
```

> Actual performance numbers should be inserted only after running the final experiment.

---

# 🛠️ Technology Stack

```mermaid
mindmap
  root((Medical AI))
    Python
      NumPy
      Pandas
      OS
    Computer Vision
      OpenCV
      ImageDataGenerator
    Deep Learning
      TensorFlow
      Keras
      CNN
    Evaluation
      Scikit-learn
      Confusion Matrix
      Classification Report
    Visualization
      Matplotlib
      Seaborn
    Development
      Jupyter Notebook
      Google Colab
      VS Code
```

---

# 📦 Dependencies

### Core

* Python
* NumPy
* Pandas
* TensorFlow
* Keras
* OpenCV

### Machine Learning

* Scikit-learn

### Visualization

* Matplotlib
* Seaborn

Example installation:

```bash
pip install -r requirements.txt
```

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/argane1/Medical-Image-Classification-CNN-.git
cd Medical-Image-Classification-CNN-
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Open:

```text
scaner.ipynb
```

The notebook can be executed through:

* Visual Studio Code
* Jupyter Notebook
* Google Colab

For faster experimentation, Google Colab GPU acceleration can be enabled.

---

# 🧪 Experimental Workflow

```mermaid
sequenceDiagram

    participant D as Dataset
    participant P as Preprocessor
    participant M as CNN Model
    participant V as Validation Set
    participant T as Test Set

    D->>P: Load Images
    P->>P: Resize & Normalize
    P->>P: Augment Training Data

    P->>M: Training Batches
    M->>M: Forward Pass
    M->>M: Backpropagation

    M->>V: Validation Predictions
    V->>M: Validation Metrics

    M->>T: Final Predictions

    T->>M: Precision / Recall / F1
```

---

# 📁 Project Structure

```text
Medical-Image-Classification-CNN/
│
├── scaner.ipynb
├── requirements.txt
├── README.md
│
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── models/
│   └── trained_model/
│
├── outputs/
│   ├── training_curves/
│   ├── confusion_matrix/
│   └── predictions/
│
└── results/
    ├── metrics.csv
    └── classification_report.txt
```

---

# 🧠 Why CNN?

Convolutional Neural Networks are particularly useful for image analysis because convolutional filters can learn hierarchical spatial patterns.

```text
Pixels
  │
  ▼
Edges
  │
  ▼
Textures
  │
  ▼
Shapes
  │
  ▼
Complex Structures
  │
  ▼
Clinical Classification
```

Early layers generally capture simple visual patterns, while deeper layers can learn increasingly complex representations.

---

# 🔬 Reproducibility

For reproducible experiments, future iterations should record:

```text
Dataset version
        ↓
Image dimensions
        ↓
Random seed
        ↓
Train / validation / test split
        ↓
Model architecture
        ↓
Hyperparameters
        ↓
Training duration
        ↓
Final metrics
```

Recommended experiment metadata:

| Parameter     | Example                           |
| ------------- | --------------------------------- |
| Image size    | 224 × 224                         |
| Batch size    | 32                                |
| Optimizer     | Adam                              |
| Learning rate | 0.001                             |
| Epochs        | 30                                |
| Loss          | Binary / Categorical Crossentropy |
| Random seed   | 42                                |

> Values above are examples and should be replaced with the actual experiment configuration.

---

# ⚠️ Limitations

This project currently has several important limitations.

### Dataset limitations

Performance depends strongly on:

* dataset quality,
* dataset size,
* class balance,
* image acquisition conditions,
* labeling quality.

### Generalization limitations

A CNN trained on one dataset may not generalize well to:

* different hospitals,
* different imaging devices,
* different patient populations,
* different acquisition protocols.

### Clinical limitations

This system is **not a clinical diagnostic tool**.

A model intended for real-world deployment would require substantially more validation, including external datasets, robustness testing, calibration, explainability, and appropriate clinical/regulatory review.

---

# 🔮 Future Improvements

The project can evolve into a considerably more advanced medical AI system.

```mermaid
flowchart LR

    A["Current CNN"] --> B["Transfer Learning"]

    B --> C["ResNet"]

    B --> D["EfficientNet"]

    B --> E["DenseNet"]

    C --> F["Model Benchmarking"]

    D --> F

    E --> F

    F --> G["Hyperparameter Optimization"]

    G --> H["Explainable AI"]

    H --> I["Grad-CAM"]

    I --> J["Model Interpretability"]

    J --> K["External Validation"]

    K --> L["Deployment"]
```

### Planned extensions

* Transfer learning with ResNet / EfficientNet / DenseNet
* Hyperparameter optimization
* Class-weighted training
* Cross-validation
* ROC-AUC and PR-AUC
* Grad-CAM visual explanations
* Model calibration
* External validation datasets
* Experiment tracking
* REST API deployment
* Docker containerization
* Cloud inference

---

# 🌐 Potential Production Architecture

A future production implementation could follow:

```mermaid
flowchart TD

    A["User / Medical System"]

    A --> B["REST API"]

    B --> C["Image Validation"]

    C --> D["Preprocessing"]

    D --> E["Trained CNN"]

    E --> F["Prediction"]

    F --> G["Confidence Score"]

    G --> H["Application Layer"]

    H --> I["Monitoring"]

    I --> J["Model Performance Tracking"]
```

This separates:

```text
Frontend
   ↓
API
   ↓
Preprocessing
   ↓
ML Model
   ↓
Prediction
   ↓
Monitoring
```

making future deployment significantly easier.

---

# 📚 Evaluation Philosophy

A strong medical AI model should not be evaluated using a single number.

The goal is to understand:

```text
                MODEL QUALITY
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
     Accuracy      Recall       Precision
        │             │             │
        └─────────────┼─────────────┘
                      ▼
                  F1 Score
                      │
                      ▼
               Error Analysis
                      │
                      ▼
              Generalization
```

The ultimate objective is therefore **reliable generalization**, not simply maximizing training accuracy.

---

# 👨‍💻 Author

**Rachid Argane**

AI / Machine Learning Engineer
Deep Learning • Computer Vision • AI Systems

GitHub:

[https://github.com/argane1](https://github.com/argane1)

---

# ⭐ Project Status

```text
🟢 Core CNN Pipeline        Implemented
🟢 Image Preprocessing      Implemented
🟢 Data Augmentation        Implemented
🟢 Model Training           Implemented
🟢 Evaluation               Implemented
🟡 Advanced Explainability  Planned
🟡 Transfer Learning        Planned
🟡 Production API           Planned
```

---

# 📜 Disclaimer

This repository is intended for **educational and research purposes only**.

The model outputs should not be used as a substitute for professional medical diagnosis, clinical judgment, or patient care.

---

<p align="center">

### ⭐ If you find this project useful, consider starring the repository.

</p>
```

