# Team Name- Pathfinder

# Team Members
# Balaji Balasubramanian
# Niranjan Niranjan

# Import libraries
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Reshape, MaxPooling2D
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator

# Load the data
train_ds = np.load('../input/ift3395-6390-quickdraw/train.npz')
test_ds = np.load('../input/ift3395-6390-quickdraw/test.npz')

# Train features and label
train_x = train_ds['arr_0']
train_y = train_ds['arr_1']

# Test features
test_x = test_ds['arr_0']

# Normalize the data
train_images = train_x / 255.0
test_images = test_x / 255.0

# One hot encoding of Training label
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = train_y.reshape(-1, 1)
y_one_hot = onehot_encoder.fit_transform(integer_encoded)

#Data augmentation
image_gen = ImageDataGenerator( rescale=1, rotation_range=15,
    width_shift_range=.15, height_shift_range=.15,
    shear_range=.15, zoom_range=.15, horizontal_flip=True)

# Reshape the train and test images
train_imagesr=  train_images.reshape( (-1,28,28,1))
test_imagesr=  test_images.reshape( (-1,28,28,1))

# Fit the data augmentation function to the training data
image_gen.fit(train_imagesr, augment=True)

# Building a Keras Sequential model (CNN)
model = Sequential()

# Adding an input layer to which training images are passed
model.add(InputLayer(input_shape=(28,28,1)))

# First convolutional layer with ReLU activation
model.add(Conv2D(kernel_size=3, strides=1, filters=32, padding='same',
                 activation='relu', name='layer_conv1'))

# Second convolutional layer with ReLU activation
model.add(Conv2D(kernel_size=3, strides=1, filters=32, padding='same',
                 activation='relu', name='layer_conv2'))

# Maxpooling layer
model.add(MaxPooling2D(pool_size=2, strides=2))


# Third convolutional layer with ReLU activation
model.add(Conv2D(kernel_size=3, strides=1, filters=64, padding='same',
                 activation='relu', name='layer_conv3'))

# Fourth convolutional layer with ReLU activation
model.add(Conv2D(kernel_size=3, strides=1, filters=64, padding='same',
                 activation='relu', name='layer_conv4'))

# Maxpooling layer
model.add(MaxPooling2D(pool_size=2, strides=2))

# Fifth convolutional layer with ReLU activation
model.add(Conv2D(kernel_size=3, strides=1, filters=128, padding='same',
                 activation='relu', name='layer_conv5'))

# Sixth convolutional layer with ReLU activation
model.add(Conv2D(kernel_size=3, strides=1, filters=128, padding='same',
                 activation='relu', name='layer_conv6'))

# Maxpooling Layer
model.add(MaxPooling2D(pool_size=2, strides=2))

# Flatten the output of the previous layer to pass them to the Dense Layer
model.add(Flatten())

# Dense Layer with ReLU-activation.
model.add(Dense(1000, activation='relu'))

# Last layer with softmax activation for classification
model.add(Dense(6, activation='softmax'))

# Nadam optimizer
optimizer = Nadam()

# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model to training data with data augmentation
model.fit_generator(image_gen.flow(train_imagesr, y_one_hot, batch_size=64),
          epochs=120, verbose=1)

# Get the prediction on test data
prediction = model.predict(test_imagesr)
prediction_max= np.argmax(prediction, axis=1)
prediction_maxs = prediction_max.astype(str)

# Creating the prediction file titled 'ml_cnn.csv'
sub =  open('ml_cnn.csv','w+')
sub.write('Id,Category\n')
for index, prediction in enumerate(prediction_maxs):
    sub.write(str(index) + ',' + prediction + '\n')
sub.close()

# The prediction file that has been created can be submitted on Kaggle.