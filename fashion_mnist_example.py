import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist


def neural_network_model(out_labels=10):
    model = Sequential()

    model.add(layers.Conv2D(16, (3, 3), padding='same', activation='relu',
                     input_shape=(28, 28, 1), data_format='channels_last'))
    model.add(layers.MaxPooling2D((2, 2), strides=2))

    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=2))

    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2), strides=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.4))
    
    model.add(layers.Dense(out_labels, activation='softmax'))

    return model


def plot_history(hist):
    # summarize history for accuracy
    plt.figure(1)
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    # summarize history for loss
    plt.figure(2)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def get_subset(images, labels, fraction):

    out_length = int(round(len(labels) * fraction))

    images_subset = images[:out_length]
    labels_subset = labels[:out_length]

    return images_subset, labels_subset


def main():

    # Load the fashion-mnist pre-shuffled train data and test data
    (images_train, labels_train), (images_test, labels_test) = fashion_mnist.load_data()

    # Choose a subset of the entire dataset
    images_train, labels_train = get_subset(images_train, labels_train, fraction=0.25)
    images_test, labels_test = get_subset(images_test, labels_test, fraction=0.25)

    # Prepare datasets 
    # normalize and reshape input images
    images_train = images_train.astype('float32') / 255
    images_train = images_train.reshape(images_train.shape[0], 28, 28, 1)
    images_test = images_test.astype('float32') / 255
    images_test = images_test.reshape(images_test.shape[0], 28, 28, 1)
    # change label numbers to one-hot vector
    labels_train = utils.to_categorical(labels_train, 10)
    labels_test = utils.to_categorical(labels_test, 10)

    # initialize a CNN model
    model = neural_network_model(out_labels=10)
    optimizer = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    history = model.fit(images_train,
                        labels_train,
                        batch_size=32,
                        epochs=10,
                        validation_split=0.2,
                        verbose=1
                        )

    # Evaluate the model on the test set
    score = model.evaluate(images_test, labels_test, verbose=0)

    # Print test accuracy
    print('\n', 'Test accuracy:', score[1])

    plot_history(history)

    # Save model weights
    model.save_weights('trained_model.h5')


if __name__ == "__main__":
    main()
