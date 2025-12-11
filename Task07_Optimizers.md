Task 07 — Optimizer Comparison Challenge

1. Objective
Compare SGD, SGD+Momentum, Adam, and AdamW on convergence and stability.

2. Code Used

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Dataset
(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1,28,28,1) / 255.0

def build_model(optimizer):
    model = keras.Sequential([
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

optimizers = {
    "SGD": keras.optimizers.SGD(0.01),
    "Momentum": keras.optimizers.SGD(0.01, momentum=0.9),
    "Adam": keras.optimizers.Adam(),
    "AdamW": keras.optimizers.AdamW()
}

histories = {}

for name, opt in optimizers.items():
    print(f"\nTraining with {name}")
    model = build_model(opt)
    hist = model.fit(x_train, y_train, epochs=10, batch_size=128,
                     validation_split=0.1, verbose=1)
    histories[name] = hist

plt.figure(figsize=(10,5))
for name in histories:
    plt.plot(histories[name].history['val_loss'], label=name)
plt.title("Optimizer Comparison — Validation Loss")
plt.legend()
plt.show()



3. Results
Training with SGD
Epoch 1/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 30s 68ms/step - accuracy: 0.6024 - loss: 1.5492 - val_accuracy: 0.9117 - val_loss: 0.3311
Epoch 2/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 27s 64ms/step - accuracy: 0.8931 - loss: 0.3843 - val_accuracy: 0.9297 - val_loss: 0.2539
Epoch 3/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 41s 63ms/step - accuracy: 0.9106 - loss: 0.3089 - val_accuracy: 0.9358 - val_loss: 0.2237
Epoch 4/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 43s 68ms/step - accuracy: 0.9193 - loss: 0.2782 - val_accuracy: 0.9447 - val_loss: 0.2021
Epoch 5/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 27s 64ms/step - accuracy: 0.9278 - loss: 0.2502 - val_accuracy: 0.9470 - val_loss: 0.1883
Epoch 6/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 27s 64ms/step - accuracy: 0.9335 - loss: 0.2267 - val_accuracy: 0.9508 - val_loss: 0.1738
Epoch 7/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 27s 64ms/step - accuracy: 0.9384 - loss: 0.2093 - val_accuracy: 0.9553 - val_loss: 0.1640
Epoch 8/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 41s 64ms/step - accuracy: 0.9433 - loss: 0.1974 - val_accuracy: 0.9595 - val_loss: 0.1485
Epoch 9/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 27s 63ms/step - accuracy: 0.9474 - loss: 0.1800 - val_accuracy: 0.9610 - val_loss: 0.1416
Epoch 10/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 41s 64ms/step - accuracy: 0.9502 - loss: 0.1699 - val_accuracy: 0.9653 - val_loss: 0.1322

Training with Momentum
Epoch 1/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 29s 65ms/step - accuracy: 0.7635 - loss: 0.8327 - val_accuracy: 0.9533 - val_loss: 0.1655
Epoch 2/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 41s 65ms/step - accuracy: 0.9505 - loss: 0.1676 - val_accuracy: 0.9668 - val_loss: 0.1135
Epoch 3/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 28s 67ms/step - accuracy: 0.9622 - loss: 0.1231 - val_accuracy: 0.9718 - val_loss: 0.0942
Epoch 4/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 27s 64ms/step - accuracy: 0.9714 - loss: 0.0915 - val_accuracy: 0.9780 - val_loss: 0.0779
Epoch 5/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 41s 64ms/step - accuracy: 0.9769 - loss: 0.0751 - val_accuracy: 0.9817 - val_loss: 0.0689
Epoch 6/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 27s 65ms/step - accuracy: 0.9814 - loss: 0.0614 - val_accuracy: 0.9793 - val_loss: 0.0733
Epoch 7/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 27s 65ms/step - accuracy: 0.9830 - loss: 0.0559 - val_accuracy: 0.9817 - val_loss: 0.0689
Epoch 8/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 40s 63ms/step - accuracy: 0.9861 - loss: 0.0452 - val_accuracy: 0.9820 - val_loss: 0.0686
Epoch 9/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 27s 63ms/step - accuracy: 0.9881 - loss: 0.0400 - val_accuracy: 0.9835 - val_loss: 0.0631
Epoch 10/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 43s 67ms/step - accuracy: 0.9897 - loss: 0.0338 - val_accuracy: 0.9852 - val_loss: 0.0587

Training with Adam
Epoch 1/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 32s 70ms/step - accuracy: 0.8700 - loss: 0.4687 - val_accuracy: 0.9803 - val_loss: 0.0812
Epoch 2/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 41s 70ms/step - accuracy: 0.9785 - loss: 0.0748 - val_accuracy: 0.9825 - val_loss: 0.0597
Epoch 3/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 40s 68ms/step - accuracy: 0.9858 - loss: 0.0477 - val_accuracy: 0.9810 - val_loss: 0.0674
Epoch 4/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 29s 68ms/step - accuracy: 0.9890 - loss: 0.0350 - val_accuracy: 0.9852 - val_loss: 0.0540
Epoch 5/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 42s 71ms/step - accuracy: 0.9924 - loss: 0.0263 - val_accuracy: 0.9842 - val_loss: 0.0572
Epoch 6/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 28s 67ms/step - accuracy: 0.9945 - loss: 0.0190 - val_accuracy: 0.9880 - val_loss: 0.0481
Epoch 7/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 29s 70ms/step - accuracy: 0.9960 - loss: 0.0139 - val_accuracy: 0.9850 - val_loss: 0.0546
Epoch 8/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 28s 66ms/step - accuracy: 0.9974 - loss: 0.0103 - val_accuracy: 0.9865 - val_loss: 0.0531
Epoch 9/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 28s 66ms/step - accuracy: 0.9980 - loss: 0.0075 - val_accuracy: 0.9848 - val_loss: 0.0718
Epoch 10/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 41s 66ms/step - accuracy: 0.9985 - loss: 0.0059 - val_accuracy: 0.9868 - val_loss: 0.0575

Training with AdamW
Epoch 1/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 30s 67ms/step - accuracy: 0.8580 - loss: 0.4823 - val_accuracy: 0.9792 - val_loss: 0.0770
Epoch 2/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 41s 68ms/step - accuracy: 0.9770 - loss: 0.0792 - val_accuracy: 0.9840 - val_loss: 0.0597
Epoch 3/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 42s 70ms/step - accuracy: 0.9852 - loss: 0.0521 - val_accuracy: 0.9860 - val_loss: 0.0530
Epoch 4/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 40s 69ms/step - accuracy: 0.9890 - loss: 0.0362 - val_accuracy: 0.9872 - val_loss: 0.0463
Epoch 5/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 29s 69ms/step - accuracy: 0.9920 - loss: 0.0265 - val_accuracy: 0.9852 - val_loss: 0.0554
Epoch 6/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 43s 74ms/step - accuracy: 0.9942 - loss: 0.0209 - val_accuracy: 0.9890 - val_loss: 0.0464
Epoch 7/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 30s 71ms/step - accuracy: 0.9959 - loss: 0.0142 - val_accuracy: 0.9888 - val_loss: 0.0470
Epoch 8/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 29s 68ms/step - accuracy: 0.9969 - loss: 0.0111 - val_accuracy: 0.9895 - val_loss: 0.0471
Epoch 9/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 40s 66ms/step - accuracy: 0.9980 - loss: 0.0085 - val_accuracy: 0.9888 - val_loss: 0.0531
Epoch 10/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 28s 66ms/step - accuracy: 0.9985 - loss: 0.0061 - val_accuracy: 0.9868 - val_loss: 0.0610
-------------------
Plots: loss and accuracy curves.

Adam converges faster and smoother than classical SGD.

4. Short Analysis
Different optimizers navigate the loss landscape differently. Momentum helps SGD escape shallow minima; Adam adapts learning rates per parameter.

5. Key Takeaway
Adam often outperforms classical optimizers in convergence speed and stability.
