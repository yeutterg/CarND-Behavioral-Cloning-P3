import csv
import os
import cv2
import sklearn
import numpy as np
from random import shuffle

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout

data_path = './data/'
log_path = data_path + 'driving_log.csv'

def process_img(image):
    """
    Preprocessing including color space conversion
    """
    # Convert to YUV color space
    print(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    return img

def generator(samples, batch_size=32):
    """
    Generator to reduce memory footprint when training
    """
    num_samples = len(samples)

    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image = cv2.imread(batch_sample[0], 1)
                angle = batch_sample[1]
                img = process_img(image)
                images.append(img)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def import_csv():
    """
    Imports the CSV file as an array of lines
    """
    lines = []
    with open(log_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    lines.pop(0) # Remove the header info from the CSV file
    return lines

def import_lrc(steer_angle):
    """
    Import the left, right, and center images from the lines of the csv file
    """
    samples = []

    data = import_csv()

    for row in data:
        steering = float(row[3])

        # center image
        center_path = data_path + row[0]
        samples.append((center_path, steering))

        # left image
        left_path = data_path + row[1]
        samples.append((left_path, steering + steer_angle))

        # right image
        right_path = data_path + row[2]
        samples.append((right_path, steering - steer_angle))

    np.random.shuffle(samples)
    return samples

ch, rw, col = 160, 320, 3  # Trimmed image format

def lenet_model():
    #LeNet model
    model = Sequential()
    model.add(Cropping2D(cropping=((70,25),(0,0)),
                         input_shape=(ch, rw, col)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    return model

def nvidia_model(): 
    #NVIDIA model
    model = Sequential()
    model.add(Cropping2D(cropping=((70, 25), (0, 0)),
                         input_shape=(ch, rw, col)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    return model

def run_model(num_epochs=3, batch_sz=32, steer_angle=0.25):
    # Import the data and split into training and validation sets
    lines = import_lrc(steer_angle)
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_sz)
    validation_generator = generator(validation_samples, batch_size=batch_sz)

    # model = lenet_model()
    model = nvidia_model()
    model.save('model.h5')

    return model.fit_generator(train_generator,
                                 samples_per_epoch=len(3*train_samples),
                                 validation_data=validation_generator,
                                 nb_val_samples=len(validation_samples),
                                 nb_epoch=num_epochs)

def plot_results(history):
    # Plot training and validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig("training_validation_loss_plot.jpg")

history = run_model(3, 128, 0.25)
plot_results(history)
exit()