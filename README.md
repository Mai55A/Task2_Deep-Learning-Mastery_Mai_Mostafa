# Deep Learning & Neural Networks — MNIST Classification Project

## Project Overview
This project explores the training and analysis of convolutional neural networks (CNNs) for handwritten digit recognition using the MNIST dataset. It focuses on understanding model behavior, generalization, and the effects of various optimization and regularization techniques.

---

## Notebook Sections
The project notebook is divided into two main parts:

### **Part 1 — Core Model Understanding & Prediction Behavior**
- **Task 1:** Deep Prediction Analysis — analyze model predictions on selected test samples.  
- **Task 2:** Custom Image Generalization Test — evaluate model performance on user-created digits.  
- **Task 3:** Epoch-Based Learning Curve Exploration — study training dynamics with different epochs.  
- **Task 4:** EarlyStopping Behavior Analysis — investigate early stopping as a regularization method.

### **Part 2 — Regularization & Optimization Mastery**
- **Task 5:** Dropout Ablation Study — examine how dropout rates affect overfitting and representation learning.  
- **Task 6:** L2 Regularization Experiment — test various L2 values and their impact on weight magnitude and generalization.  
- **Task 7:** Optimizer Comparison Challenge — compare SGD, SGD with momentum, Adam, and AdamW in terms of convergence and stability.  
- **Task 8:** Batch Size & Gradient Noise Experiment — analyze the effect of batch size on training noise and generalization.  
- **Task 9:** Activation Function Swap — explore ReLU, Tanh, Softsign, and GELU effects on gradient flow and performance.  
- **Task 10:** Weight Inspection & Model Capacity Analysis — study learned weights and the impact of high model capacity.

---

## How to Run the Notebook
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <your-project-folder>
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
jupyter notebook notebook.ipynb

All plots, results, and predictions are saved in the results/ folder.
Sample Results
Task 1 — Prediction Analysis

Task 5 — Dropout Ablation

Task 7 — Optimizer Comparison

All additional plots and outputs are saved in the results/ folder for review

📘 README — ML Assignment (Part 1 & Part 2)
Digit Classification using Neural Networks — Comprehensive Study
📍 Part 1 — Fundamental MNIST Experiments

This part explores the core pipeline of training and evaluating a neural network using the MNIST handwritten digits dataset. It includes dataset loading, model building, training, evaluation, confusion matrix computation, and generalization tests.

### ✅ Task 1 — MNIST Loading & Dataset Exploration

Goal:
Load the MNIST dataset and perform basic dataset exploration.

What was done:

Loaded training and testing splits using keras.datasets.mnist.load_data().

Normalized pixel values to the range [0,1].

Visualized a sample of images.

Examined dataset dimensions and value distributions.

What we learned:

MNIST contains 60,000 training images and 10,000 test images.

Normalization improves convergence and stabilizes training.

Visual inspection helps verify dataset correctness.

### ✅ Task 2 — Custom Image Generalization Test

Goal:
Evaluate how well the trained MNIST model generalizes to real or external images.

What was done:

Attempted to load a custom digit image.

If the image was missing, generated a synthetic digit using random strokes.

Converted any loaded image to:

28×28 resolution

grayscale

inverted color (white digit on black background)

normalized floating-point tensor

Performed prediction and compared probability scores.

What we learned:

Models trained on MNIST can classify digits even when noise or small distortions are added.

If the file does not exist, the system automatically generates a synthetic fallback image for testing.

External images require careful preprocessing to match MNIST format.

### ✅ Task 3 — Neural Network Implementation & Training

Goal:
Build and train a fully connected neural network classifier.

Architecture:

Flatten → Dense(128, ReLU) → Dense(10, Softmax)

What was done:

Compiled the model with Adam optimizer and Sparse Categorical Crossentropy.

Trained with validation monitoring.

Saved training curves for both loss and accuracy.

What we learned:

ReLU activation accelerates learning.

Adam optimizer converges quickly.

Validation accuracy indicates generalization strength and overfitting.

### ✅ Task 4 — Confusion Matrix & Detailed Error Analysis

Goal:
Evaluate misclassifications to understand model weaknesses.

What was done:

Used predictions to create a confusion matrix.

Visualized it using seaborn heatmap.

Highlighted which digits are most commonly confused (e.g., 4 vs 9, 3 vs 5).

What we learned:

Confusion matrices reveal patterns that accuracy hides.

Certain digit pairs share similar shapes, so misclassification is expected.

Visualization helps guide model improvements.

🟦 Part 2 — Regularization & Optimization Mastery

This part focuses on improving generalization, tuning optimization behavior, analyzing learning dynamics, and understanding model capacity.

### 🧪 Task 5 — Dropout Ablation Study

Goal:
Evaluate how dropout affects overfitting.

Experiments:

Dropout = 0.0

Dropout = 0.1

Dropout = 0.3

What was done:

Train 3 separate models.

Plot training vs. validation loss for each.

Compare overfitting severity.

Explain the role of dropout.

What we learned:

No Dropout: Highest overfitting, lowest validation performance.

Dropout 0.1: Reduced overfitting, best balance.

Dropout 0.3: Too much noise, slower convergence.

📌 Dropout prevents neuron co-adaptation, forcing the network to learn redundant yet robust representations.

### 🧪 Task 6 — L2 Regularization Experiment

Goal:
Determine how weight decay affects training and generalization.

Values tested:

0.0001

0.001

0.01

What was done:

Added kernel_regularizer=keras.regularizers.l2(lambda) to the Dense layers.

Tracked training and validation losses.

Observed how L2 pulls weights toward smaller magnitudes.

What we learned:

Smaller L2: Slightly smoother curves, minor improvement.

Medium L2 (0.001): Best validation loss (ideal weight shrinking).

Large L2 (0.01): Underfitting due to excessive penalty.

📌 L2 promotes small, smooth weights → better generalization & less variance.

### ⚙️ Task 7 — Optimizer Comparison Challenge

Goal:
Train identical models using different optimizers.

Optimizers tested:

SGD

SGD + Momentum

Adam

AdamW

What was done:

Trained four models.

Recorded training/validation accuracy and loss.

Compared convergence.

What we learned:

SGD: Slow, unstable convergence.

SGD + Momentum: Faster & smoother, still sensitive to learning rate.

Adam: Fastest convergence, highest accuracy, best stability.

AdamW: Similar to Adam but better generalization due to decoupled weight decay.

📌 Adam navigates the loss landscape efficiently using adaptive learning rates per parameter.

### 📊 Task 8 — Batch Size & Gradient Noise Experiment

Goal:
Understand how batch size affects optimization.

Batch sizes tested:

8

32

128

What was found:

Batch 8: Noisy gradients → better generalization but slower training.

Batch 32: Best overall stability vs. performance (recommended).

Batch 128: Smooth curves but easily overfits → worse generalization.

📌 Small batches introduce “gradient noise” that helps escape sharp local minima.

### 🔄 Task 9 — Activation Function Swap

Goal:
Evaluate how different activation functions affect learning dynamics.

Replacements tested:

Tanh

Softsign

GELU

Analysis:

Tanh: Smooth but suffers from vanishing gradients.

Softsign: Similar to tanh but lighter saturation.

GELU: Best performance; widely used in Transformers because it activates neurons probabilistically.

📌 ReLU remains popular due to simplicity & strong gradient flow.

### 🧠 Task 10 — Weight Inspection & Model Capacity Analysis

Goal:
Understand model capacity by inspecting learned parameters.

What was done:

Extracted first Dense layer weights using:

w, b = model.layers[1].get_weights()
print(w.shape)


Discussed parameter counts and overfitting risk.

What we learned:

First Dense layer contains hundreds of thousands of parameters.

High parameter count → high capacity → high risk of memorizing training data.

Regularization techniques (Dropout, L2, EarlyStopping) help mitigate this.
