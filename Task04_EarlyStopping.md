Task 04 — EarlyStopping Behavior Analysis

1. Objective
Investigate EarlyStopping as a form of regularization.

2. Code Used


# task4_early_stopping.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ------------------
# Load data
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

callback = keras.callbacks.EarlyStopping(
    patience=3,
    restore_best_weights=True
)

model = build_cnn()

history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=128,
    callbacks=[callback],
    verbose=1
)

print("\nTraining stopped at epoch:", len(history.history["loss"]))



3. Results
Epoch 1/50
422/422 ━━━━━━━━━━━━━━━━━━━━ 44s 101ms/step - accuracy: 0.8381 - loss: 0.5325 - val_accuracy: 0.9835 - val_loss: 0.0604
Epoch 2/50
422/422 ━━━━━━━━━━━━━━━━━━━━ 82s 100ms/step - accuracy: 0.9790 - loss: 0.0683 - val_accuracy: 0.9880 - val_loss: 0.0512
Epoch 3/50
422/422 ━━━━━━━━━━━━━━━━━━━━ 44s 103ms/step - accuracy: 0.9850 - loss: 0.0462 - val_accuracy: 0.9887 - val_loss: 0.0427
Epoch 4/50
422/422 ━━━━━━━━━━━━━━━━━━━━ 44s 103ms/step - accuracy: 0.9900 - loss: 0.0322 - val_accuracy: 0.9887 - val_loss: 0.0438
Epoch 5/50
422/422 ━━━━━━━━━━━━━━━━━━━━ 43s 102ms/step - accuracy: 0.9920 - loss: 0.0250 - val_accuracy: 0.9890 - val_loss: 0.0377
Epoch 6/50
422/422 ━━━━━━━━━━━━━━━━━━━━ 83s 103ms/step - accuracy: 0.9947 - loss: 0.0176 - val_accuracy: 0.9893 - val_loss: 0.0417
Epoch 7/50
422/422 ━━━━━━━━━━━━━━━━━━━━ 81s 100ms/step - accuracy: 0.9954 - loss: 0.0148 - val_accuracy: 0.9862 - val_loss: 0.0504
Epoch 8/50
422/422 ━━━━━━━━━━━━━━━━━━━━ 44s 104ms/step - accuracy: 0.9963 - loss: 0.0106 - val_accuracy: 0.9892 - val_loss: 0.0452

Training stopped at epoch: 8




4. Short Analysis
EarlyStopping prevents overfitting by halting training when val_loss plateaus. Increasing patience delays stopping. Using SGD may change the epoch at which stopping occurs.

5. Key Takeaway
EarlyStopping acts as indirect regularization by avoiding unnecessary weight updates.
