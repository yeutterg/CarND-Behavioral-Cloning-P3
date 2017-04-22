import csv
import os
import cv2
import sklearn
import numpy as np
from random import shuffle

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def import_csv():
    """
    Imports the CSV file
    """
    lines = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    lines.pop(0) # Remove the header info from the CSV file
    return lines

def process_img(image):
    """
    Preprocessing including color space conversion
    """
    # Convert to YUV color space
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
                for i in range(3):
                    name = './data/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    angle = float(batch_sample[3])
                    img = process_img(image)
                    images.append(img)
                    angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Import the data and split into training and validation sets
lines = import_csv()
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout

ch, row, col = 160, 320, 3  # Trimmed image format

#LeNet model
#model = Sequential()
#model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
#model.add(Cropping2D(cropping=((70,25),(0,0))))
#model.add(Convolution2D(6,5,5,activation="relu"))
#model.add(MaxPooling2D())
#model.add(Convolution2D(6,5,5,activation="relu"))
#model.add(MaxPooling2D())
#model.add(Flatten())
#model.add(Dense(120))
#model.add(Dense(84))
#model.add(Dense(1))

#NVIDIA model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Cropping2D(cropping=((70, 25), (0, 0)),
                     input_shape=(ch, row, col)))
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

fitgen = model.fit_generator(train_generator,
                             samples_per_epoch=len(3*train_samples),
                             validation_data=validation_generator,
                             nb_val_samples=len(validation_samples),
                             nb_epoch=5)

model.save('model.h5')

# Plot training and validation loss
plt.plot(fitgen.history['loss'])
plt.plot(fitgen.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig("training_validation_loss_plot.jpg")

exit()
