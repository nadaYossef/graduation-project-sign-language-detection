import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import math  # mathematical functions
import cv2  # OpenCV for image processing
import matplotlib.pyplot as plt  # data visualization

# TensorFlow and Keras for building and training neural networks
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, AveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # for image augmentation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler

# Sklearn for data processing and evaluation
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedKFold, cross_val_score
from sklearn.metrics import f1_score, classification_report
from sklearn.datasets import make_classification
from sklearn.utils import class_weight

# Keras for loading models
from keras.models import load_model

import pickle

def load_data(file_path):
    """
    Load training data from a CSV file.

    Args:
        file_path (str): The file path to the CSV file.

    Returns:
        pd.DataFrame: The loaded training data.
    """
    return pd.read_csv(file_path)

def preprocess_data(data):
    """
    Preprocess the data by normalizing inputs and encoding targets.

    Args:
        data (pd.DataFrame): The training data.

    Returns:
        tuple: Normalized inputs and encoded targets.
    """
    targets = data['label']
    inputs = data.drop(['label'], axis=1)
    inputs = np.array(inputs)
    inputs = np.reshape(inputs, (inputs.shape[0], 28, 28, 1))
    inputs = inputs / 255.0
    targets = to_categorical(targets)
    return inputs, targets

def create_data_generator():
    """
    Create an ImageDataGenerator for data augmentation.

    Returns:
        ImageDataGenerator: The data generator for image augmentation.
    """
    return ImageDataGenerator(rotation_range=10, zoom_range=0.1, width_shift_range=0.1,
                               height_shift_range=0.1, shear_range=0.1, horizontal_flip=True)

def step_decay(epoch):
    """
    Define a step decay learning rate schedule.

    Args:
        epoch (int): The current epoch number.

    Returns:
        float: The updated learning rate.
    """
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

def my_model():
    """
    Build and compile the CNN model.

    Returns:
        Sequential: The compiled Keras Sequential model.
    """
    classifier = Sequential()
    classifier.add(Conv2D(1024, (3, 3), input_shape=(28, 28, 1), padding='same', activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Conv2D(512, (5, 5), padding='same', activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Conv2D(256, (7, 7), padding='same', activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Flatten())
    classifier.add(Dense(units=1024, activation='relu'))
    classifier.add(Dropout(0.2))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units=512, activation='relu'))
    classifier.add(Dropout(0.2))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units=256, activation='relu'))
    classifier.add(Dropout(0.2))
    classifier.add(BatchNormalization())
    classifier.add(Dense(units=25, activation="softmax"))
    classifier.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=0.0, nesterov=False),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
    return classifier

def train_model(model, x_train, y_train, x_test, y_test, batch_size=32, epochs=20):
    """
    Train the model using k-fold cross-validation.

    Args:
        model (Sequential): The Keras model to train.
        x_train (np.ndarray): The training input data.
        y_train (np.ndarray): The training target data.
        x_test (np.ndarray): The testing input data.
        y_test (np.ndarray): The testing target data.
        batch_size (int): The batch size for training.
        epochs (int): The number of epochs for training.

    Returns:
        float: The mean accuracy from cross-validation.
    """
    training_data_array = np.array(x_train)
    labels_array = np.array(y_train)
    kfold = RepeatedKFold(n_splits=5, n_repeats=1)
    cvscores = []

    for train_index, test_index in kfold.split(training_data_array, labels_array.argmax(1)):
        x_train_fold, x_test_fold = training_data_array[train_index], training_data_array[test_index]
        y_train_fold, y_test_fold = labels_array[train_index], labels_array[test_index]
        data_gen = create_data_generator()
        train_generator = data_gen.flow(x_train_fold, y_train_fold)

        class_weights = class_weight.compute_class_weight(class_weight="balanced",
                                                          classes=np.unique(y_train_fold.argmax(1)),
                                                          y=y_train_fold.argmax(1))
        class_weights = dict(enumerate(class_weights))

        lrate = LearningRateScheduler(step_decay)
        model.fit(x_train_fold, y_train_fold, batch_size=batch_size, epochs=epochs, verbose=1,
                  validation_data=(x_test_fold, y_test_fold))
        scores = model.evaluate(x_test_fold, y_test_fold, verbose=0)
        cvscores.append(scores[1] * 100)

    return np.mean(cvscores), np.std(cvscores)

def save_model(model, file_name):
    """
    Save the trained model to a file.

    Args:
        model (Sequential): The Keras model to save.
        file_name (str): The file name to save the model.
    """
    model.save(file_name)

def load_trained_model(file_name):
    """
    Load a trained Keras model from a file.

    Args:
        file_name (str): The file name to load the model from.

    Returns:
        Sequential: The loaded Keras model.
    """
    return load_model(file_name)

def main():
    # Load data
    train_data = load_data('sign_mnist_train.csv')
    inputs, targets = preprocess_data(train_data)

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.1, random_state=0)

    # Create and train model
    model = my_model()
    mean_accuracy, std_accuracy = train_model(model, inputs, targets, x_test, y_test)

    # Save model
    save_model(model, "sign-language.h5")

    # Print model summary and accuracy
    print("Model accuracy: %.2f%% (+/- %.2f%%)" % (mean_accuracy, std_accuracy))
    loaded_model = load_trained_model("sign-languaget.h5")
    print(loaded_model.summary())

if __name__ == "__main__":
    main()

    # Save the necessary variables and model using pickle
    with open('model_data.pkl', 'wb') as f:
        pickle.dump((inputs, targets), f)
