Task 06 — L2 Regularization Experiment

1. Objective
Examine the effect of L2 weight regularization on generalization.

2. Code Used

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt

# Same dataset
(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1,28,28,1) / 255.0

def build_model(l2_value):
    model = keras.Sequential([
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_value)),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

l2_values = [0.0001, 0.001, 0.01]
histories = {}

for reg in l2_values:
    print(f"\nTraining with L2 = {reg}")
    model = build_model(reg)
    history = model.fit(x_train, y_train, epochs=10, batch_size=128,
                        validation_split=0.1, verbose=1)
    histories[reg] = history

# Plot
plt.figure(figsize=(10,5))
for reg in l2_values:
    plt.plot(histories[reg].history['val_loss'], label=f"L2={reg}")
plt.title("Validation Loss with Different L2 Regularizations")
plt.legend()
plt.show()


3. Results

Training with L2 = 0.0001
Epoch 1/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 30s 67ms/step - accuracy: 0.8681 - loss: 0.4985 - val_accuracy: 0.9797 - val_loss: 0.1142
Epoch 2/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 41s 68ms/step - accuracy: 0.9759 - loss: 0.1162 - val_accuracy: 0.9830 - val_loss: 0.0935
Epoch 3/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 40s 66ms/step - accuracy: 0.9822 - loss: 0.0933 - val_accuracy: 0.9818 - val_loss: 0.0905
Epoch 4/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 41s 67ms/step - accuracy: 0.9860 - loss: 0.0800 - val_accuracy: 0.9848 - val_loss: 0.0854
Epoch 5/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 28s 67ms/step - accuracy: 0.9881 - loss: 0.0721 - val_accuracy: 0.9855 - val_loss: 0.0821
Epoch 6/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 28s 66ms/step - accuracy: 0.9900 - loss: 0.0640 - val_accuracy: 0.9860 - val_loss: 0.0841
Epoch 7/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 29s 69ms/step - accuracy: 0.9909 - loss: 0.0600 - val_accuracy: 0.9868 - val_loss: 0.0767
Epoch 8/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 28s 67ms/step - accuracy: 0.9916 - loss: 0.0573 - val_accuracy: 0.9857 - val_loss: 0.0765
Epoch 9/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 28s 67ms/step - accuracy: 0.9921 - loss: 0.0553 - val_accuracy: 0.9862 - val_loss: 0.0786
Epoch 10/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 28s 65ms/step - accuracy: 0.9947 - loss: 0.0478 - val_accuracy: 0.9862 - val_loss: 0.0808

Training with L2 = 0.001
Epoch 1/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 30s 68ms/step - accuracy: 0.8628 - loss: 0.6520 - val_accuracy: 0.9722 - val_loss: 0.1974
Epoch 2/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 29s 68ms/step - accuracy: 0.9653 - loss: 0.2057 - val_accuracy: 0.9728 - val_loss: 0.1722
Epoch 3/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 28s 66ms/step - accuracy: 0.9723 - loss: 0.1671 - val_accuracy: 0.9782 - val_loss: 0.1484
Epoch 4/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 29s 69ms/step - accuracy: 0.9758 - loss: 0.1502 - val_accuracy: 0.9712 - val_loss: 0.1597
Epoch 5/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 40s 67ms/step - accuracy: 0.9773 - loss: 0.1410 - val_accuracy: 0.9777 - val_loss: 0.1475
Epoch 6/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 28s 66ms/step - accuracy: 0.9788 - loss: 0.1361 - val_accuracy: 0.9832 - val_loss: 0.1185
Epoch 7/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 30s 70ms/step - accuracy: 0.9815 - loss: 0.1212 - val_accuracy: 0.9790 - val_loss: 0.1364
Epoch 8/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 28s 66ms/step - accuracy: 0.9818 - loss: 0.1197 - val_accuracy: 0.9835 - val_loss: 0.1175
Epoch 9/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 28s 66ms/step - accuracy: 0.9825 - loss: 0.1170 - val_accuracy: 0.9763 - val_loss: 0.1385
Epoch 10/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 28s 65ms/step - accuracy: 0.9828 - loss: 0.1096 - val_accuracy: 0.9837 - val_loss: 0.1122

Training with L2 = 0.01
Epoch 1/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 32s 71ms/step - accuracy: 0.8469 - loss: 1.0846 - val_accuracy: 0.9462 - val_loss: 0.3391
Epoch 2/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 28s 67ms/step - accuracy: 0.9384 - loss: 0.3495 - val_accuracy: 0.9540 - val_loss: 0.2781
Epoch 3/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 42s 70ms/step - accuracy: 0.9467 - loss: 0.3142 - val_accuracy: 0.9578 - val_loss: 0.2895
Epoch 4/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 40s 67ms/step - accuracy: 0.9543 - loss: 0.2823 - val_accuracy: 0.9697 - val_loss: 0.2195
Epoch 5/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 29s 68ms/step - accuracy: 0.9546 - loss: 0.2690 - val_accuracy: 0.9710 - val_loss: 0.2158
Epoch 6/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 40s 67ms/step - accuracy: 0.9598 - loss: 0.2390 - val_accuracy: 0.9690 - val_loss: 0.1957
Epoch 7/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 28s 67ms/step - accuracy: 0.9642 - loss: 0.2136 - val_accuracy: 0.9728 - val_loss: 0.1953
Epoch 8/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 28s 67ms/step - accuracy: 0.9665 - loss: 0.2031 - val_accuracy: 0.9710 - val_loss: 0.1929
Epoch 9/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 41s 68ms/step - accuracy: 0.9659 - loss: 0.2045 - val_accuracy: 0.9760 - val_loss: 0.1782
Epoch 10/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 41s 68ms/step - accuracy: 0.9695 - loss: 0.1848 - val_accuracy: 0.9743 - val_loss: 0.1729
4. Short Analysis
L2 reduces weight magnitude, discouraging overfitting. Smaller weights help improve generalization and produce smoother validation curves.

5. Key Takeaway
Regularization balances model complexity and generalization.
