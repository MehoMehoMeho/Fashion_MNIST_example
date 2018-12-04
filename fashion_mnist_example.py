import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils

from keras.datasets import fashion_mnist


def neural_network_model(out_labels=10):
    model = Sequential()

    model.add(Conv2D(16, (3, 3), padding='same', activation='relu',
                     input_shape=(28, 28, 1), data_format='channels_last'))
    model.add(MaxPooling2D((2, 2), strides=2))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=2))

    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2), strides=2))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.35))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.35))
    model.add(Dense(out_labels, activation='softmax'))

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


def main():

    # Load the fashion-mnist pre-shuffled train data and test data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Prepare datasets
    # This step contains normalization and reshaping of input.
    # For output, it is important to change number to one-hot vector.
    x_train = x_train.astype('float32') / 255
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    # Split train data to train and validation data
    train_len = 50000

    images_train = x_train[:train_len]
    images_validation = x_train[train_len:]
    labels_train = y_train[:train_len]
    labels_validation = y_train[train_len:]

    model = neural_network_model(out_labels=10)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(images_train,
                        labels_train,
                        batch_size=64,
                        epochs=30,
                        validation_data=(images_validation, labels_validation),
                        verbose=2
                        )

    # Evaluate the model on test set
    score = model.evaluate(x_test, y_test, verbose=0)

    # Print test accuracy
    print('\n', 'Test accuracy:', score[1])

    plot_history(history)


if __name__ == "__main__":
    main()
