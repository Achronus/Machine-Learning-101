# View CPU and GPU devices
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Convolutional Neural Networks

#----------------------------------------
# Part 1 - Building the CNN
#----------------------------------------
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu', data_format = 'channels_last' ))
# Step 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2) ))

# Step 1 - Convolution (Secondary)
classifier.add(Convolution2D(32, (3, 3), activation = 'relu', data_format = 'channels_last' ))
# Step 2 - Max Pooling (Secondary)
classifier.add(MaxPooling2D(pool_size = (2, 2) ))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
# Hidden Layer
classifier.add(Dense(units = 128, activation = 'relu'))
# Output Layer - uses sigmoid for 'Binary' (two values) and uses 'Softmax' for 3+ values
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#----------------------------------------
# Part 2 - Fitting the CNN to the images
#----------------------------------------
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size = (64, 64),
        batch_size = 32,
        class_mode = 'binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size = (64, 64),
        batch_size = 32,
        class_mode = 'binary')

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

with tf.device('/GPU:0'):
    classifier.fit_generator(
            training_set,
            steps_per_epoch = 8000,
            epochs = 1,
            validation_data = test_set,
            validation_steps = 2000)


#----------------------------------------
# Part 3 - Making new predictions
#----------------------------------------
import numpy as np
from keras.preprocessing import image
test_image = image.load_img(path = 'dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = classifier.predict(test_image)
indices = training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
