import numpy as np
import matplotlib.pyplot as plt
from fashion_mnist_example import neural_network_model
from tensorflow.keras.datasets import fashion_mnist


def visualize_predicted_classes(images, labels, predicted_classes):
        
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

    for ind, (image, label, predicted_class) in enumerate(zip(images, labels, predicted_classes)):

        plt.subplot(7, 7, ind+1)
        plt.imshow(255 - np.squeeze(image),  cmap='gray')
        plt.axis('off')

        if label == predicted_class:
            plt.title(fashion_mnist_labels[label])
        else:
            plt.title(fashion_mnist_labels[predicted_class] + "!=" + fashion_mnist_labels[label], color='red')

    plt.tight_layout()
    plt.show()


_, (images_test, labels_test) = fashion_mnist.load_data()

# normalize and reshape input images
images_test = images_test.astype('float32') / 255
images_test = images_test.reshape(images_test.shape[0], 28, 28, 1)

# choose a small subset
images = images_test[:49]
labels = labels_test[:49]

# define the model and load the trained weights
model = neural_network_model()
model.load_weights('trained_model.h5')

# get predictions on the test iamges
predictions = model.predict(images)

# find predicted classes (class index of the max predicted values)
predicted_classes = predictions.argmax(axis=-1)

visualize_predicted_classes(images, labels, predicted_classes)