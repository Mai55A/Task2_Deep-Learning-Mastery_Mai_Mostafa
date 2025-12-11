Task 09 — Activation Function Swap (ReLU vs Tanh vs GELU)

1. Objective
Compare different activations on gradient flow and model performance.

2. Code Used

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1,28,28,1)/255.0

def build_model(activation_fn):
    model = keras.Sequential([
        layers.Conv2D(32,3,activation=activation_fn),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation=activation_fn),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

activations = ["tanh", "softsign", "gelu"]
histories = {}

for act in activations:
    print(f"\nTraining with activation = {act}")
    model = build_model(act)
    hist = model.fit(x_train, y_train, epochs=10,
                     batch_size=128, validation_split=0.1, verbose=1)
    histories[act] = hist

plt.figure(figsize=(10,5))
for act in activations:
    plt.plot(histories[act].history["val_loss"], label=act)
plt.legend()
plt.title("Activation Function Comparison")
plt.show()


3. Results
   Training with activation = tanh
Epoch 1/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 34s 76ms/step - accuracy: 0.8621 - loss: 0.4511 - val_accuracy: 0.9688 - val_loss: 0.1103
Epoch 2/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 40s 73ms/step - accuracy: 0.9695 - loss: 0.1091 - val_accuracy: 0.9777 - val_loss: 0.0783
Epoch 3/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 44s 79ms/step - accuracy: 0.9820 - loss: 0.0630 - val_accuracy: 0.9832 - val_loss: 0.0648
Epoch 4/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 32s 75ms/step - accuracy: 0.9902 - loss: 0.0384 - val_accuracy: 0.9833 - val_loss: 0.0567
Epoch 5/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 31s 73ms/step - accuracy: 0.9936 - loss: 0.0257 - val_accuracy: 0.9853 - val_loss: 0.0530
Epoch 6/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 33s 78ms/step - accuracy: 0.9969 - loss: 0.0168 - val_accuracy: 0.9860 - val_loss: 0.0477
Epoch 7/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 41s 77ms/step - accuracy: 0.9977 - loss: 0.0121 - val_accuracy: 0.9868 - val_loss: 0.0472
Epoch 8/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 30s 70ms/step - accuracy: 0.9982 - loss: 0.0094 - val_accuracy: 0.9882 - val_loss: 0.0480
Epoch 9/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 41s 72ms/step - accuracy: 0.9991 - loss: 0.0062 - val_accuracy: 0.9873 - val_loss: 0.0471
Epoch 10/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 31s 73ms/step - accuracy: 0.9997 - loss: 0.0039 - val_accuracy: 0.9878 - val_loss: 0.0465

Training with activation = softsign
Epoch 1/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 32s 71ms/step - accuracy: 0.8624 - loss: 0.4969 - val_accuracy: 0.9677 - val_loss: 0.1249
Epoch 2/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 30s 72ms/step - accuracy: 0.9601 - loss: 0.1390 - val_accuracy: 0.9768 - val_loss: 0.0852
Epoch 3/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 30s 72ms/step - accuracy: 0.9765 - loss: 0.0829 - val_accuracy: 0.9797 - val_loss: 0.0734
Epoch 4/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 30s 70ms/step - accuracy: 0.9842 - loss: 0.0581 - val_accuracy: 0.9838 - val_loss: 0.0635
Epoch 5/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 40s 69ms/step - accuracy: 0.9886 - loss: 0.0435 - val_accuracy: 0.9840 - val_loss: 0.0581
Epoch 6/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 30s 70ms/step - accuracy: 0.9925 - loss: 0.0320 - val_accuracy: 0.9857 - val_loss: 0.0526
Epoch 7/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 41s 71ms/step - accuracy: 0.9943 - loss: 0.0240 - val_accuracy: 0.9860 - val_loss: 0.0523
Epoch 8/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 41s 70ms/step - accuracy: 0.9955 - loss: 0.0198 - val_accuracy: 0.9853 - val_loss: 0.0518
Epoch 9/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 29s 68ms/step - accuracy: 0.9970 - loss: 0.0138 - val_accuracy: 0.9848 - val_loss: 0.0535
Epoch 10/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 42s 71ms/step - accuracy: 0.9978 - loss: 0.0119 - val_accuracy: 0.9865 - val_loss: 0.0496

Training with activation = gelu
Epoch 1/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 57s 130ms/step - accuracy: 0.8431 - loss: 0.5396 - val_accuracy: 0.9755 - val_loss: 0.0948
Epoch 2/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 54s 128ms/step - accuracy: 0.9718 - loss: 0.0948 - val_accuracy: 0.9822 - val_loss: 0.0625
Epoch 3/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 84s 133ms/step - accuracy: 0.9840 - loss: 0.0533 - val_accuracy: 0.9823 - val_loss: 0.0606
Epoch 4/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 79s 125ms/step - accuracy: 0.9884 - loss: 0.0395 - val_accuracy: 0.9865 - val_loss: 0.0503
Epoch 5/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 54s 129ms/step - accuracy: 0.9922 - loss: 0.0266 - val_accuracy: 0.9867 - val_loss: 0.0557
Epoch 6/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 80s 124ms/step - accuracy: 0.9944 - loss: 0.0204 - val_accuracy: 0.9880 - val_loss: 0.0516
Epoch 7/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 52s 124ms/step - accuracy: 0.9961 - loss: 0.0136 - val_accuracy: 0.9845 - val_loss: 0.0584
Epoch 8/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 82s 125ms/step - accuracy: 0.9976 - loss: 0.0102 - val_accuracy: 0.9883 - val_loss: 0.0582
Epoch 9/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 81s 123ms/step - accuracy: 0.9986 - loss: 0.0065 - val_accuracy: 0.9895 - val_loss: 0.0500
Epoch 10/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 53s 125ms/step - accuracy: 0.9989 - loss: 0.0050 - val_accuracy: 0.9868 - val_loss: 0.0574
---------------
ReLU: fast convergence, risk of dead neurons low.

Tanh/Softsign: potential vanishing gradients.

GELU: smooth, effective for deep architectures.

4. Short Analysis
Activation choice affects gradient propagation. GELU is preferred in transformers; ReLU remains effective for CNNs/MLPs.

5. Key Takeaway
Activation function selection balances gradient flow and computational efficiency.
