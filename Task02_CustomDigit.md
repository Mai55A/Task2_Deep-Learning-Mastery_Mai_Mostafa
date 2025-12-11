Task 02 — Custom Image Generalization Test

1. Objective
Evaluate model generalization on a user-created or synthetic digit image not seen during training.

2. Code Used
# task1_task2_synthetic.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random
import os
import json
from PIL import Image, ImageDraw

# -------------------------
# reproducibility seed
# -------------------------
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

print("TensorFlow version:", tf.__version__)

# -------------------------
# 1) load MNIST
# -------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)
num_classes = 10

# -------------------------
# 2) build CNN model
# -------------------------
def build_cnn():
    model = keras.Sequential([
        layers.Input(shape=(28,28,1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPool2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# -------------------------
# 3) train model quickly
# -------------------------
model = build_cnn()
print("Training for 5 epochs (quick) ...")
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=1)

# -------------------------
# Task 1 — Deep Prediction Analysis
# -------------------------
indices = [10, 123, 999]
samples = x_test[indices]
true_labels = y_test[indices]

probs = model.predict(samples)           # softmax probabilities
pred_labels = np.argmax(probs, axis=1)   # predicted classes

print("\n=== Task 1 results ===")
for i, idx in enumerate(indices):
    plt.figure(figsize=(2.2,2.2))
    plt.imshow(samples[i].squeeze(), cmap='gray')
    plt.title(f"Index {idx} | Pred: {pred_labels[i]} | True: {true_labels[i]}")
    plt.axis('off')
    plt.show()
    print(f"Index {idx} -> Predicted: {pred_labels[i]}  True: {true_labels[i]}")
    print("Softmax top probabilities (top 5):")
    top5 = np.argsort(probs[i])[::-1][:5]
    for k in top5:
        print(f"  class {k}: {probs[i][k]:.4f}")
    print("-"*36)

# save Task 1 results
out_task1 = {
    "indices": indices,
    "pred_labels": pred_labels.tolist(),
    "true_labels": true_labels.tolist(),
    "probs": probs.tolist()
}
with open("task1_results.json", "w") as f:
    json.dump(out_task1, f)
print("Saved Task 1 results to task1_results.json")

# -------------------------
# Task 2 — Custom Synthetic Image
# -------------------------
print("\n\n=== Task 2: Custom Synthetic Image Test ===")

# create a simple synthetic digit (like "3")
img_size = 140
img = Image.new("L", (img_size, img_size), color=255)  # white background
draw = ImageDraw.Draw(img)
draw.line([(40,30),(100,30),(100,70),(40,70)], width=10)
draw.line([(40,70),(100,70),(100,110),(40,110)], width=10)
# add some random noise
for _ in range(200):
    x = random.randint(35,105)
    y = random.randint(25,115)
    img.putpixel((x,y), random.randint(0,40))

# resize to 28x28
img_small = img.resize((28,28), resample=Image.Resampling.LANCZOS)
arr = np.array(img_small).astype("float32")
arr = 255 - arr          # invert to match MNIST white digit on black
arr = arr / 255.0
processed_image = arr.reshape(1,28,28,1)

# display the synthetic image
plt.figure(figsize=(2,2))
plt.imshow(arr, cmap='gray')
plt.title("Task 2: Synthetic Image (28x28)")
plt.axis('off')
plt.show()

# run prediction
pred = model.predict(processed_image)
pred_label = int(np.argmax(pred))
probs_list = pred.flatten().tolist()

print(f"Predicted Label for synthetic image: {pred_label}")
print("Softmax Probabilities (all classes):")
for i, p in enumerate(probs_list):
    print(f"  class {i}: {p:.4f}")

# save Task 2 results
out_task2 = {
    "predicted_label": pred_label,
    "probabilities": probs_list
}
with open("task2_results.json", "w") as f:
    json.dump(out_task2, f)
print("Saved Task 2 results to task2_results.json")

processed_image = create_synthetic_digit()  # 28x28 normalized
pred = model.predict(processed_image)
pred_label = np.argmax(pred)


3. Results
=== Task 2: Custom Synthetic Image Test ===

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 106ms/step
Predicted Label for synthetic image: 3
Softmax Probabilities (all classes):
  class 0: 0.0000
  class 1: 0.0000
  class 2: 0.0000
  class 3: 0.9999
  class 4: 0.0000
  class 5: 0.0000
  class 6: 0.0000
  class 7: 0.0000
  class 8: 0.0000
  class 9: 0.0000
Saved Task 2 results to task2_results.json
Predicted Label: 3

Softmax probabilities for all classes printed.

Plot: the processed synthetic digit.

4. Short Analysis
Distribution shift and minor noise can affect predictions. Model successfully generalizes if high-level shape matches MNIST features. Lack of augmentation may limit robustness for unusual strokes.

5. Key Takeaway
Neural networks can generalize to new inputs but may fail if input distribution differs significantly from training data.
