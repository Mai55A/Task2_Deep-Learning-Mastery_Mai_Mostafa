Part 2 — Regularization & Optimization Mastery
Task 05 — Dropout Ablation Study

1. Objective
Evaluate how different Dropout rates affect overfitting and representation robustness.

2. Code Used

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1,28,28,1)/255.0
x_test  = x_test.reshape(-1,28,28,1)/255.0

def build_model(dropout_rate):
    model = keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

dropouts = [0.0, 0.1, 0.3]
histories = {}

for d in dropouts:
    print(f"\nTraining with Dropout = {d}")
    model = build_model(d)
    history = model.fit(x_train, y_train, epochs=10, batch_size=128,
                        validation_split=0.1, verbose=1)
    histories[d] = history

# Plot
plt.figure(figsize=(10,5))
for d in dropouts:
    plt.plot(histories[d].history['val_loss'], label=f'Dropout {d}')
plt.title("Validation Loss for Different Dropout Rates")
plt.legend()
plt.show()



3. Results
Training with Dropout = 0.0
/usr/local/lib/python3.12/dist-packages/keras/src/layers/convolutional/base_conv.py:113: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
Epoch 1/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 31s 71ms/step - accuracy: 0.8644 - loss: 0.4848 - val_accuracy: 0.9765 - val_loss: 0.0852
Epoch 2/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 28s 66ms/step - accuracy: 0.9752 - loss: 0.0837 - val_accuracy: 0.9863 - val_loss: 0.0543
Epoch 3/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 29s 69ms/step - accuracy: 0.9857 - loss: 0.0492 - val_accuracy: 0.9877 - val_loss: 0.0496
Epoch 4/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 43s 73ms/step - accuracy: 0.9899 - loss: 0.0353 - val_accuracy: 0.9887 - val_loss: 0.0448
Epoch 5/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 38s 66ms/step - accuracy: 0.9929 - loss: 0.0242 - val_accuracy: 0.9880 - val_loss: 0.0480
Epoch 6/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 28s 66ms/step - accuracy: 0.9948 - loss: 0.0186 - val_accuracy: 0.9887 - val_loss: 0.0467
Epoch 7/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 41s 66ms/step - accuracy: 0.9969 - loss: 0.0122 - val_accuracy: 0.9880 - val_loss: 0.0451
Epoch 8/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 29s 68ms/step - accuracy: 0.9977 - loss: 0.0098 - val_accuracy: 0.9892 - val_loss: 0.0504
Epoch 9/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 28s 66ms/step - accuracy: 0.9983 - loss: 0.0075 - val_accuracy: 0.9870 - val_loss: 0.0590
Epoch 10/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 29s 68ms/step - accuracy: 0.9980 - loss: 0.0073 - val_accuracy: 0.9872 - val_loss: 0.0578

Training with Dropout = 0.1
Epoch 1/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 29s 67ms/step - accuracy: 0.8568 - loss: 0.4755 - val_accuracy: 0.9755 - val_loss: 0.0896
Epoch 2/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 28s 66ms/step - accuracy: 0.9728 - loss: 0.0924 - val_accuracy: 0.9817 - val_loss: 0.0645
Epoch 3/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 29s 68ms/step - accuracy: 0.9837 - loss: 0.0551 - val_accuracy: 0.9853 - val_loss: 0.0559
Epoch 4/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 41s 68ms/step - accuracy: 0.9867 - loss: 0.0425 - val_accuracy: 0.9865 - val_loss: 0.0501
Epoch 5/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 30s 71ms/step - accuracy: 0.9910 - loss: 0.0305 - val_accuracy: 0.9898 - val_loss: 0.0442
Epoch 6/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 39s 66ms/step - accuracy: 0.9934 - loss: 0.0231 - val_accuracy: 0.9898 - val_loss: 0.0470
Epoch 7/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 26s 62ms/step - accuracy: 0.9941 - loss: 0.0190 - val_accuracy: 0.9903 - val_loss: 0.0456
Epoch 8/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 27s 64ms/step - accuracy: 0.9952 - loss: 0.0156 - val_accuracy: 0.9887 - val_loss: 0.0478
Epoch 9/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 27s 65ms/step - accuracy: 0.9966 - loss: 0.0115 - val_accuracy: 0.9893 - val_loss: 0.0499
Epoch 10/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 28s 65ms/step - accuracy: 0.9965 - loss: 0.0103 - val_accuracy: 0.9907 - val_loss: 0.0454

Training with Dropout = 0.3
Epoch 1/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 29s 66ms/step - accuracy: 0.8505 - loss: 0.5175 - val_accuracy: 0.9778 - val_loss: 0.0818
Epoch 2/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 28s 66ms/step - accuracy: 0.9704 - loss: 0.1014 - val_accuracy: 0.9828 - val_loss: 0.0581
Epoch 3/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 40s 64ms/step - accuracy: 0.9796 - loss: 0.0676 - val_accuracy: 0.9880 - val_loss: 0.0464
Epoch 4/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 28s 65ms/step - accuracy: 0.9845 - loss: 0.0528 - val_accuracy: 0.9868 - val_loss: 0.0461
Epoch 5/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 40s 64ms/step - accuracy: 0.9874 - loss: 0.0413 - val_accuracy: 0.9887 - val_loss: 0.0420
Epoch 6/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 41s 63ms/step - accuracy: 0.9887 - loss: 0.0360 - val_accuracy: 0.9878 - val_loss: 0.0429
Epoch 7/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 42s 65ms/step - accuracy: 0.9902 - loss: 0.0313 - val_accuracy: 0.9887 - val_loss: 0.0437
Epoch 8/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 28s 67ms/step - accuracy: 0.9923 - loss: 0.0249 - val_accuracy: 0.9882 - val_loss: 0.0413
Epoch 9/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 42s 69ms/step - accuracy: 0.9930 - loss: 0.0217 - val_accuracy: 0.9903 - val_loss: 0.0373
Epoch 10/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 36s 86ms/step - accuracy: 0.9940 - loss: 0.0197 - val_accuracy: 0.9880 - val_loss: 0.0446
Plots: training vs. validation loss.

Higher dropout reduces overfitting at the cost of slightly slower convergence.

4. Short Analysis
Dropout prevents neurons from co-adapting, forcing the network to learn more robust representations.

5. Key Takeaway
Moderate dropout improves generalization without drastically reducing performance.
