from keras.layers import RandomRotation, RandomZoom, BatchNormalization
from tensorflow.python.keras.saving.save import load_model
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

model = load_model("self_model/best_train_model.h5", custom_objects={'RandomRotation': RandomRotation, 'RandomZoom': RandomZoom, 'BatchNormalization': BatchNormalization})

def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.convert('L')
    image = image.resize((28, 28))
    image = np.array(image)
    image = image.astype('float32') / 255
    image = image.reshape(1, 28, 28, 1)
    return image

def plot_predictions(image):
    predictions = model.predict(image)
    plt.figure(figsize=(3, 3))
    pred_label = np.argmax(predictions[0])
    plt.imshow(image[0].reshape(28, 28), cmap='gray')
    color = 'blue'
    plt.title(f"Pred: {pred_label}", color=color)
    plt.axis('off')
    plt.show()

image_path = "yolo/result_img_for_predict/digit_9.png"
new_image = load_and_preprocess_image(image_path)
plot_predictions(new_image)
