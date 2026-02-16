---
# Testing Overfitting in Neural Networks using PyTorch (Fashion-MNIST)

## Project Overview

This project explores overfitting behavior in deep neural networks using the Fashion-MNIST dataset. Multiple neural network architectures were implemented in PyTorch to observe how model complexity affects training loss and generalization.

_The goal of this experiment is to:

* Understand how deeper networks can overfit
* Compare shallow vs deep architectures
* Analyze the effect of Batch Normalization
* Observe training loss patterns across epochs

---

## Dataset

Dataset used: Fashion-MNIST (CSV format)

* 28x28 grayscale images (flattened to 784 features)
* 10 clothing categories
* First column: Label
* Remaining columns: Pixel values (0–255)

Data preprocessing steps:

* Train-test split (80% train, 20% test)
* Pixel normalization (divided by 255)
* Conversion to PyTorch tensors
* Custom Dataset and DataLoader implementation

---

## Project Structure

```
overfitting-pytorch-fashion-mnist/
│
├── overfitting_experiment.py
└── README.md
```

---

## Technologies Used

* Python
* PyTorch
* pandas
* scikit-learn
* matplotlib

---

## Implementation Details

### 1. Custom Dataset Class

A custom PyTorch Dataset class was created to:

* Convert features to float32 tensors
* Convert labels to long tensors
* Enable batch loading using DataLoader

---

## Experiments Conducted

### Experiment 1: Moderate Network with Batch Normalization

Architecture:

* Linear(784 → 128)
* BatchNorm
* ReLU
* Linear(128 → 64)
* BatchNorm
* ReLU
* Linear(64 → 10)

Optimizer: SGD
Learning rate: 0.1
Epochs: 80

Observation:

* Training loss steadily decreased
* Batch normalization stabilized learning
* Reduced internal covariate shift

---

### Experiment 2: Deep Network (High Capacity Model)

Architecture:

* Linear(784 → 128)
* Linear(128 → 110)
* Linear(110 → 100)
* Linear(100 → 80)
* Linear(80 → 60)
* Linear(60 → 40)
* Linear(40 → 20)
* Linear(20 → 10)

All layers followed by ReLU activation.

Optimizer: SGD
Learning rate: 0.01
Epochs: 80

Observation:

* Extremely low training loss
* Indicates potential overfitting
* High-capacity model memorizes training data

This experiment demonstrates how deeper networks can overfit when regularization is absent.

---

### Experiment 3: Smaller Network (Controlled Capacity)

Architecture:

* Linear(784 → 128)
* ReLU
* Linear(128 → 44)
* ReLU
* Linear(44 → 10)

Optimizer: SGD
Learning rate: 0.1
Epochs: 25

Observation:

* Slower and more stable convergence
* Reduced risk of overfitting
* More balanced training behavior

---

## Training Setup

* Loss Function: CrossEntropyLoss
* Optimizer: Stochastic Gradient Descent (SGD)
* Device: GPU if available, otherwise CPU
* Batch sizes: 32 and 42 (varied across experiments)

---

## Key Concepts Demonstrated

1. Model Capacity and Overfitting
   Deeper models with more parameters tend to memorize training data.

2. Batch Normalization
   Helps stabilize training and improve convergence.

3. Learning Rate Sensitivity
   Higher learning rates can speed up training but may destabilize convergence.

4. Importance of Architecture Design
   Simply increasing depth does not guarantee better generalization.

---

## Lessons Learned

* Lower training loss does not guarantee better generalization.
* Deep models require regularization (Dropout, Weight Decay, Early Stopping).
* Batch normalization improves training stability.
* Monitoring both training and validation metrics is essential to detect overfitting.

---

## Future Improvements

* Add validation accuracy tracking
* Plot training vs validation loss curves
* Implement Dropout regularization
* Add L2 weight decay
* Implement early stopping
* Use Adam optimizer for comparison
* Evaluate final test accuracy

---

## How to Run

1. Install dependencies:

```
pip install torch pandas scikit-learn matplotlib
```

2. Update the dataset path inside the script.

3. Run:

```
python overfitting_experiment.py
```

---

## Conclusion

This project provides a practical demonstration of how neural network depth and architectural choices influence overfitting. By comparing multiple model configurations, we observe how increasing complexity can drastically reduce training loss while increasing the risk of poor generalization.

This experiment strengthens understanding of model design principles in deep learning.

---

