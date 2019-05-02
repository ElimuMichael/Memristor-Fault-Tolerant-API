from keras.datasets import mnist, cifar10, fashion_mnist
from keras.utils import to_categorical
import keras.backend as K
import json


def get_data(data_type):
    batch_size = 0
    num_classes = 0
    input_shape = ()

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Convert the Targets to Categorical values
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    # Reshape Training and Test Datasets
    x_train = X_train.reshape(-1, 28, 28, 1)
    x_test = X_test.reshape(-1, 28, 28, 1)

    if data_type == 'MNIST' or data_type == 'mnist':
        batch_size = 64
        input_shape = (28, 28, 1)
        num_classes = 10
        x_train = x_train
        x_test = x_test
        y_train_cat = y_train_cat
        y_test_cat = y_test_cat

    elif data_type == 'FashionMNIST' or data_type == 'fashionmnist':
        # Number of classes
        num_classes = 10
        batch_size = 128
        img_rows, img_cols = (28, 28)

        # Load the training and test data from keras
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        # Reshape the data as required by the backend
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        # Scale the pixel intensities
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        x_train /= 255
        x_test /= 255

        # Change the y values to categorical values
        y_train_cat = to_categorical(y_train, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)

    dataset = {
        'input_shape': input_shape,
        'num_classes': num_classes,
        'batch_size': batch_size,
        'x_train': x_train,
        'y_train': y_train_cat,
        'x_test': x_test,
        'y_test': y_test_cat
    }

    return dataset


def read_hist():
    fileName = 'history/history.txt'
    with open(fileName, 'r+') as f:
        content = f.read()

    return content
