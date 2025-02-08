import tensorflow as tf
import deeplake
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2


ds = deeplake.load('hub://activeloop/fer2013-train')
ds.summary()
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

images = ds['images'][:].numpy()
labels = ds['labels'][:].numpy()
images = images.astype('float32') / 255.0


images = images[..., None]
train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='sigmoid', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='sigmoid'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation='sigmoid'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(7, activation='softmax')
])

model.load_weights('my_model.weights.h5')

def predict_and_display(image_path):
    # Load the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(5,5))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title("Original Image")
    plt.show()

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    resized_img = cv2.resize(gray_img, (48, 48))

    img_array = resized_img.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions)
    predicted_class_name = emotion_labels[predicted_class_idx]

    plt.figure(figsize=(5,5))
    plt.imshow(resized_img, cmap='gray')
    plt.axis('off')
    plt.title(f"Predicted: {predicted_class_name}")
    plt.show()
