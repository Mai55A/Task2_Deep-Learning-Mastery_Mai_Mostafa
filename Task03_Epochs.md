Task 03 — Epoch-Based Learning Curve Exploration

1. Objective
Study training and validation dynamics over different epoch counts to identify overfitting.

2. Code Used
# task3_learning_curves.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ------------------
# Load MNIST
# ------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32")/255.0
x_test  = x_test.astype("float32")/255.0
x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)

# ------------------
# Build model
# ------------------
def build_cnn():
    model = keras.Sequential([
        layers.Input(shape=(28,28,1)),
        layers.Conv2D(32,3,activation="relu"),
        layers.MaxPool2D(),
        layers.Conv2D(64,3,activation="relu"),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(128,activation="relu"),
        layers.Dense(10,activation="softmax")
    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

epochs_list = [5, 10, 20]
histories = {}

for ep in epochs_list:
    print(f"\nTraining for {ep} epochs...")
    model = build_cnn()
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=ep,
        batch_size=128,
        verbose=1
    )
    histories[ep] = history

# ------------------
# Plot Curves
# ------------------
plt.figure(figsize=(14,6))

# Loss curves
plt.subplot(1,2,1)
for ep in epochs_list:
    plt.plot(histories[ep].history["loss"], label=f"train loss ({ep} ep)")
    plt.plot(histories[ep].history["val_loss"], label=f"val loss ({ep} ep)")
plt.title("Loss vs Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Accuracy curves
plt.subplot(1,2,2)
for ep in epochs_list:
    plt.plot(histories[ep].history["accuracy"], label=f"train acc ({ep} ep)")
    plt.plot(histories[ep].history["val_accuracy"], label=f"val acc ({ep} ep)")
plt.title("Accuracy vs Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

breifly
history_5 = model.fit(x_train, y_train, epochs=5, validation_split=0.1)
history_10 = model.fit(x_train, y_train, epochs=10, validation_split=0.1)
history_20 = model.fit(x_train, y_train, epochs=20, validation_split=0.1)


3. Results
Plots: loss vs. val_loss, accuracy vs. val_accuracy.

Observed slight overfitting at 20 epochs.

4. Short Analysis
Adam optimizer accelerated convergence. Overfitting appears when val_loss stagnates while training loss continues decreasing. Monitoring curves helps choose optimal epochs.

5. Key Takeaway
Tracking loss and accuracy per epoch is crucial to prevent overfitting.
