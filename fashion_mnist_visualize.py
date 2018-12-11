import math
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist


def visualize_classes(images, labels, predictions=None):
        
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

    plt.figure(figsize=(12, 12))

    for ind, (image, label) in enumerate(zip(images, labels)):

        plt.subplot(7, 7, ind+1)
        plt.imshow(255 - image,  cmap='gray')
        plt.axis('off')

        predicted_class = label

        plt.title(fashion_mnist_labels[label])

    plt.tight_layout()
    plt.show()


def main():
    # Load the fashion-mnist pre-shuffled train data and test data
    (images_train, labels_train), _ = fashion_mnist.load_data()

    print(images_train.shape)

    images = images_train[:49]
    labels = labels_train[:49]

    visualize_classes(images, labels)


if __name__ == "__main__":
    main()
