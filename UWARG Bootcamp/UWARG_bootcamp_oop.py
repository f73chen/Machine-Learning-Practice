# import keras packages and the CIFAR-10 dataset
import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# loads "a dataset of 50,000 32x32 color training images and 10,000 test images, labeled over 10 categories"
TRAIN_SIZE = 50000             # 50,000 images in training set
TEST_SIZE = 10000              # 10,000 images in test/validation set
IMG_ROWS, IMG_COLS = 32, 32    # each image is 32x32 pixels
NUM_CHANNELS = 3               # 3 channels for 3 colours RGB
NUM_CLASSES = 10               # 10 categories

BATCH_SIZE = 64                # learns from batches of 64 images instead of the full training set
EPOCHS = 1                     # number of iterations over the entire training set (accuracy_graph.png was based on 400 epochs)

class CNNAgent:
    def get_data(self):
        (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()    
        xTrain = xTrain.astype('float32')/255.0
        xTest = xTest.astype('float32')/255.0    
        yTrain = keras.utils.to_categorical(yTrain, NUM_CLASSES)
        yTest = keras.utils.to_categorical(yTest, NUM_CLASSES)
        return xTrain, xTest, yTrain, yTest

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(IMG_ROWS, IMG_COLS, NUM_CHANNELS)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(10, activation='softmax'))

        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=opt,
                    metrics=['accuracy'])
        return model

    def __init__(self):
        self.model = self.create_model()
        self.xTrain, self.xTest, self.yTrain, self.yTest = self.get_data()

    def train(self):
        self.history = self.model.fit(self.xTrain, self.yTrain,
                epochs=EPOCHS, batch_size = BATCH_SIZE,
                verbose=0,
                validation_data=(self.xTest, self.yTest))

    def print_stats(self):
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

# start off with model and x, y train & test data in self.
agent = CNNAgent()
agent.train()
agent.print_stats()
