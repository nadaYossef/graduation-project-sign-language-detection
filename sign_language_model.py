import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.utils import class_weight

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    targets = data['label']
    inputs = data.drop(['label'], axis=1)
    inputs = np.array(inputs).reshape(-1, 28, 28, 1) / 255.0
    targets = to_categorical(targets)
    return inputs, targets

def create_data_generator():
    return ImageDataGenerator(rotation_range=10, zoom_range=0.1, 
                               width_shift_range=0.1, height_shift_range=0.1, 
                               shear_range=0.1, horizontal_flip=True)

def step_decay(epoch):
    initial_lrate = 0.001  # Adjusted learning rate
    drop = 0.5
    epochs_drop = 10.0
    return initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))

def my_model():
    classifier = Sequential()
    classifier.add(Input(shape=(28, 28, 1)))  # Use Input layer as the first layer
    classifier.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(BatchNormalization())
    classifier.add(Dropout(0.3))

    classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(BatchNormalization())
    classifier.add(Dropout(0.4))

    classifier.add(Flatten())
    classifier.add(Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    classifier.add(Dropout(0.5))
    classifier.add(BatchNormalization())
    
    classifier.add(Dense(units=25, activation="softmax"))
    classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                       loss='categorical_crossentropy', 
                       metrics=['accuracy'])
    return classifier

def train_model(model, x_train, y_train, x_test, y_test, batch_size=32, epochs=50):
    data_gen = create_data_generator()
    train_generator = data_gen.flow(x_train, y_train, batch_size=batch_size)

    class_weights = class_weight.compute_class_weight(class_weight="balanced", 
                                                      classes=np.unique(y_train.argmax(1)), 
                                                      y=y_train.argmax(1))
    class_weights = dict(enumerate(class_weights))

    lrate = LearningRateScheduler(step_decay)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min')
    
    model.fit(train_generator, validation_data=(x_test, y_test),
              epochs=epochs, verbose=1, class_weight=class_weights,
              callbacks=[lrate, early_stopping, checkpoint])

    scores = model.evaluate(x_test, y_test, verbose=0)
    return scores[1] * 100, scores[0] * 100

def save_model(model, file_name):
    model.save(file_name)

def load_trained_model(file_name):
    return tf.keras.models.load_model(file_name)

def main():
    train_data = load_data('sign_mnist_train.csv')
    inputs, targets = preprocess_data(train_data)

    x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.1, random_state=0)

    model = my_model()
    accuracy, loss = train_model(model, x_train, y_train, x_test, y_test)

    save_model(model, "sign-language.keras")
    print("Model accuracy: %.2f%%, loss: %.2f%%" % (accuracy, loss))
    loaded_model = load_trained_model("sign-language.keras") 
    print(loaded_model.summary())

if __name__ == "__main__":
    main()
