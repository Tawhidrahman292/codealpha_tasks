# mnist_cnn.py
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Step 1 — Load MNIST dataset
# -----------------------------
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Normalize the images
x_train = x_train.reshape((60000, 28, 28, 1)) / 255.0
x_test = x_test.reshape((10000, 28, 28, 1)) / 255.0

# -----------------------------
# Step 2 — Build CNN model
# -----------------------------
model = models.Sequential([
    layers.Input(shape=(28,28,1)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# -----------------------------
# Step 3 — Train the model
# -----------------------------
history = model.fit(x_train, y_train, epochs=3, validation_split=0.1)

# -----------------------------
# Step 4 — Evaluate on test data
# -----------------------------
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

# -----------------------------
# Step 5 — Visualize predictions
# -----------------------------
prob_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = prob_model.predict(x_test)

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i].reshape(28,28)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    color = 'green' if predicted_label == true_label else 'red'
    plt.xlabel(f"{predicted_label} ({true_label})", color=color)
    plt.show()

# Plot first 5 test images with predictions
for i in range(5):
    plot_image(i, predictions, y_test, x_test)

# -----------------------------
# Step 6 — Save the trained model
# -----------------------------
model.save("mnist_cnn_model.h5")
print("Model saved as mnist_cnn_model.h5")
