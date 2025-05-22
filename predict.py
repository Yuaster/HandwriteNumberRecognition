from keras.layers import RandomRotation, RandomZoom, BatchNormalization
from tensorflow.python.keras.saving.save import load_model
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

model = load_model("self_model/best_train_model.h5", custom_objects={'RandomRotation': RandomRotation, 'RandomZoom': RandomZoom, 'BatchNormalization': BatchNormalization})


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


# if __name__ == '__main__':
#     image_path = "yolo_about/result_img_for_predict/digit_9.png"
#     new_image = load_and_preprocess_image(image_path)
#     plot_predictions(new_image)
