Task 08 — Batch Size & Gradient Noise Experiment

1. Objective
Study impact of batch size on gradient noise and training dynamics.

2. Code Used

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1,28,28,1)/255.0

def build_model():
    model = keras.Sequential([
        layers.Conv2D(32,3,activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128,activation='relu'),
        layers.Dense(10,activation='softmax')
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

batch_sizes = [8, 32, 128]
histories = {}

for b in batch_sizes:
    print(f"\nTraining with batch size = {b}")
    model = build_model()
    hist = model.fit(x_train, y_train, batch_size=b,
                     epochs=10, validation_split=0.1, verbose=1)
    histories[b] = hist

plt.figure(figsize=(10,5))
for b in batch_sizes:
    plt.plot(histories[b].history["val_loss"], label=f"batch={b}")
plt.legend()
plt.title("Batch Size Validation Loss")
plt.show()



3. Results
   Training with batch size = 8
Epoch 1/10
6750/6750 ━━━━━━━━━━━━━━━━━━━━ 104s 15ms/step - accuracy: 0.9260 - loss: 0.2471 - val_accuracy: 0.9813 - val_loss: 0.0664
Epoch 2/10
6750/6750 ━━━━━━━━━━━━━━━━━━━━ 98s 15ms/step - accuracy: 0.9843 - loss: 0.0500 - val_accuracy: 0.9873 - val_loss: 0.0456
Epoch 3/10
6750/6750 ━━━━━━━━━━━━━━━━━━━━ 147s 15ms/step - accuracy: 0.9921 - loss: 0.0252 - val_accuracy: 0.9872 - val_loss: 0.0531
Epoch 4/10
6750/6750 ━━━━━━━━━━━━━━━━━━━━ 102s 15ms/step - accuracy: 0.9950 - loss: 0.0159 - val_accuracy: 0.9873 - val_loss: 0.0550
Epoch 5/10
6750/6750 ━━━━━━━━━━━━━━━━━━━━ 100s 15ms/step - accuracy: 0.9961 - loss: 0.0126 - val_accuracy: 0.9880 - val_loss: 0.0628
Epoch 6/10
6750/6750 ━━━━━━━━━━━━━━━━━━━━ 111s 16ms/step - accuracy: 0.9974 - loss: 0.0082 - val_accuracy: 0.9878 - val_loss: 0.0753
Epoch 7/10
6750/6750 ━━━━━━━━━━━━━━━━━━━━ 106s 16ms/step - accuracy: 0.9982 - loss: 0.0055 - val_accuracy: 0.9890 - val_loss: 0.0763
Epoch 8/10
6750/6750 ━━━━━━━━━━━━━━━━━━━━ 104s 15ms/step - accuracy: 0.9979 - loss: 0.0066 - val_accuracy: 0.9870 - val_loss: 0.0939
Epoch 9/10
6750/6750 ━━━━━━━━━━━━━━━━━━━━ 145s 16ms/step - accuracy: 0.9982 - loss: 0.0057 - val_accuracy: 0.9887 - val_loss: 0.0781
Epoch 10/10
6750/6750 ━━━━━━━━━━━━━━━━━━━━ 103s 15ms/step - accuracy: 0.9987 - loss: 0.0042 - val_accuracy: 0.9872 - val_loss: 0.0883

Training with batch size = 32
Epoch 1/10
1688/1688 ━━━━━━━━━━━━━━━━━━━━ 43s 24ms/step - accuracy: 0.8979 - loss: 0.3381 - val_accuracy: 0.9802 - val_loss: 0.0699
Epoch 2/10
1688/1688 ━━━━━━━━━━━━━━━━━━━━ 43s 26ms/step - accuracy: 0.9819 - loss: 0.0613 - val_accuracy: 0.9833 - val_loss: 0.0651
Epoch 3/10
1688/1688 ━━━━━━━━━━━━━━━━━━━━ 80s 24ms/step - accuracy: 0.9876 - loss: 0.0381 - val_accuracy: 0.9845 - val_loss: 0.0527
Epoch 4/10
1688/1688 ━━━━━━━━━━━━━━━━━━━━ 81s 24ms/step - accuracy: 0.9935 - loss: 0.0220 - val_accuracy: 0.9875 - val_loss: 0.0490
Epoch 5/10
1688/1688 ━━━━━━━━━━━━━━━━━━━━ 40s 24ms/step - accuracy: 0.9951 - loss: 0.0150 - val_accuracy: 0.9867 - val_loss: 0.0492
Epoch 6/10
1688/1688 ━━━━━━━━━━━━━━━━━━━━ 42s 25ms/step - accuracy: 0.9964 - loss: 0.0111 - val_accuracy: 0.9880 - val_loss: 0.0541
Epoch 7/10
1688/1688 ━━━━━━━━━━━━━━━━━━━━ 41s 24ms/step - accuracy: 0.9980 - loss: 0.0071 - val_accuracy: 0.9888 - val_loss: 0.0526
Epoch 8/10
1688/1688 ━━━━━━━━━━━━━━━━━━━━ 43s 25ms/step - accuracy: 0.9986 - loss: 0.0046 - val_accuracy: 0.9852 - val_loss: 0.0800
Epoch 9/10
1688/1688 ━━━━━━━━━━━━━━━━━━━━ 42s 25ms/step - accuracy: 0.9979 - loss: 0.0057 - val_accuracy: 0.9853 - val_loss: 0.0743
Epoch 10/10
1688/1688 ━━━━━━━━━━━━━━━━━━━━ 44s 26ms/step - accuracy: 0.9981 - loss: 0.0051 - val_accuracy: 0.9882 - val_loss: 0.0599

Training with batch size = 128
Epoch 1/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 33s 75ms/step - accuracy: 0.8634 - loss: 0.4955 - val_accuracy: 0.9803 - val_loss: 0.0738
Epoch 2/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 30s 71ms/step - accuracy: 0.9770 - loss: 0.0738 - val_accuracy: 0.9855 - val_loss: 0.0583
Epoch 3/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 30s 70ms/step - accuracy: 0.9862 - loss: 0.0455 - val_accuracy: 0.9860 - val_loss: 0.0527
Epoch 4/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 29s 68ms/step - accuracy: 0.9890 - loss: 0.0354 - val_accuracy: 0.9880 - val_loss: 0.0460
Epoch 5/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 43s 72ms/step - accuracy: 0.9929 - loss: 0.0226 - val_accuracy: 0.9850 - val_loss: 0.0524
Epoch 6/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 40s 70ms/step - accuracy: 0.9947 - loss: 0.0185 - val_accuracy: 0.9877 - val_loss: 0.0513
Epoch 7/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 30s 70ms/step - accuracy: 0.9959 - loss: 0.0129 - val_accuracy: 0.9847 - val_loss: 0.0594
Epoch 8/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 29s 69ms/step - accuracy: 0.9968 - loss: 0.0119 - val_accuracy: 0.9882 - val_loss: 0.0550
Epoch 9/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 43s 74ms/step - accuracy: 0.9978 - loss: 0.0081 - val_accuracy: 0.9877 - val_loss: 0.0499
Epoch 10/10
422/422 ━━━━━━━━━━━━━━━━━━━━ 31s 73ms/step - accuracy: 0.9987 - loss: 0.0065 - val_accuracy: 0.9878 - val_loss: 0.0580

-------------------------------
Smaller batches → noisier gradients, slower but sometimes better generalization.

Larger batches → smoother loss curves, faster convergence, slightly worse generalization.

4. Short Analysis
Gradient noise from small batches can help escape local minima. Large batches reduce stochasticity, affecting exploration.

5. Key Takeaway
Batch size controls trade-off between convergence speed and generalization.
