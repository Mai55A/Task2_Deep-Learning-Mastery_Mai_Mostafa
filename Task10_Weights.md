Task 10 — Weight Inspection & Model Capacity Analysis

1. Objective
Analyze weight magnitude and model capacity from first Dense layer.

2. Code Used

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1,28,28,1)/255.0

model = keras.Sequential([
    layers.Conv2D(32,3,activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1)

w, b = model.layers[3].get_weights()
print("Weight shape for Dense layer:", w.shape)
print("Bias shape:", b.shape)


3. Results

Epoch 1/3
422/422 ━━━━━━━━━━━━━━━━━━━━ 32s 70ms/step - accuracy: 0.8720 - loss: 0.4747 - val_accuracy: 0.9772 - val_loss: 0.0822
Epoch 2/3
422/422 ━━━━━━━━━━━━━━━━━━━━ 29s 68ms/step - accuracy: 0.9767 - loss: 0.0819 - val_accuracy: 0.9833 - val_loss: 0.0570
Epoch 3/3
422/422 ━━━━━━━━━━━━━━━━━━━━ 28s 66ms/step - accuracy: 0.9853 - loss: 0.0500 - val_accuracy: 0.9853 - val_loss: 0.0532
Weight shape for Dense layer: (5408, 128)
Bias shape: (128,)
----------------------
Shape example: (1600, 128) → 204,800 parameters.

High model capacity can overfit small datasets.

4. Short Analysis
Dropout, L2, and EarlyStopping mitigate overfitting risk by controlling weight magnitude or halting training early.

5. Key Takeaway
Understanding model capacity is crucial for balancing performance and generalization.
