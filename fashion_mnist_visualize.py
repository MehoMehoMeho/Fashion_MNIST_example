import math
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist


def explore_dataset():

    # Load the fashion-mnist pre-shuffled train data and test data
    (images_train, labels_train), _ = fashion_mnist.load_data()

    # fashion mnist label name
    fashion_mnist_labels = np.array([
        'T-shirt/top',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle boot'])

    images = images_train[:49].copy()
    labels = labels_train[:49].copy()

    plt.figure(figsize=(12, 12))

    for ind, (image, label) in enumerate(zip(images, labels)):

        plt.subplot(7, 7, ind+1)
        plt.imshow(255 - image,  cmap='gray')
        plt.axis('off')

        predicted_class = label

        if label == predicted_class:
            plt.title(fashion_mnist_labels[predicted_class])
        else:
            plt.title(fashion_mnist_labels[predicted_class] + "!=" + fashion_mnist_labels[label], color='red')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    explore_dataset()
