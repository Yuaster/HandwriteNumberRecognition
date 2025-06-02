from keras.layers import RandomRotation, RandomZoom, BatchNormalization
from tensorflow.python.keras.saving.save import load_model
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json

with open('config.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

model = load_model(data["model"], custom_objects={'RandomRotation': RandomRotation, 'RandomZoom': RandomZoom, 'BatchNormalization': BatchNormalization})


def load_and_preprocess_image(image, is_pil_image=False):
    if is_pil_image:
        pil_image = image
    else:
        pil_image = Image.open(image)
    pil_image = pil_image.convert('L')
    pil_image = pil_image.resize((28, 28))
    image_array = np.array(pil_image)
    image_array = image_array.astype('float32') / 255
    image_array = image_array.reshape(1, 28, 28, 1)

    return image_array

def plot_predictions(image):
    predictions = model.predict(image)
    plt.figure(figsize=(3, 3))
    pred_label = np.argmax(predictions[0])
    plt.imshow(image[0].reshape(28, 28), cmap='gray')
    color = 'blue'
    plt.title(f"Pred: {pred_label}", color=color)
    plt.axis('off')
    plt.show()

def get_predictions(image):
    predictions = model.predict(image)
    pred_label = np.argmax(predictions[0])
    return pred_label

if __name__ == '__main__':
    img = load_and_preprocess_image("number_box_about/result_img_for_predict/digit_3.png")
    plot_predictions(img)
