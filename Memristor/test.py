# Import Models
from keras import layers
from keras.models import model_from_json
from keras import optimizers
from keras.models import Sequential
from keras.layers import Input, Dropout, BatchNormalization, Dense, Conv2D, MaxPool2D, AveragePooling2D, Flatten, MaxPooling2D, Activation, concatenate, GlobalMaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils, np_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from keras.regularizers import l1, l2
import util
import numpy as np


def Lenet5():
    # Build The Model
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(5, 5),
                     input_shape=(28, 28, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(16, kernel_size=(5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(120, kernel_size=(5, 5)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(84))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model

    #####################################
    #     MEMRISTOR WEIGHTS MODEL       #
    #####################################


# Visualize Results
#     for res in tools:
#         plt.style.use('ggplot')
#         plt.figure(figsize=(10, 10))
#         for i in range(6):
#             plt.subplot(1, 6, i+1)
#             plt.imshow(X_test[i, :,:].reshape((28,28)), cmap='gray')
#             plt.gca().get_xaxis().set_ticks([])
#             plt.gca().get_yaxis().set_ticks([])
#             plt.xlabel('Pred: %d' % res[i])
#         plt.show()
    # print(Accuracy[1], Accuracy_retrained[1])
