# Team Name- Pathfinder

# Team Members
# Balaji Balasubramanian
# Niranjan Niranjan

# Import libraries
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
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

# We will build an ensemble of 4 CNN models

# CNN model 1
# Data augmentation
image_gen1 = ImageDataGenerator(
    rescale=1,
    rotation_range=15,
    width_shift_range=.15,
    height_shift_range=.15,
    shear_range=.15,
    zoom_range=.15,
    horizontal_flip=True,
    )

# Reshape the train and test images
train_imagesr=  train_images.reshape( (-1,28,28,1))
test_imagesr=  test_images.reshape( (-1,28,28,1))

# Train and test split
X_tr1, X_eval1, y_tr1 ,y_eval1 = train_test_split(train_imagesr, 
                y_one_hot,test_size=0.1,shuffle=True,random_state=0,stratify=train_y)

# Fit the data augmentation function to the training data
image_gen1.fit(X_tr1, augment=True)

# Building a Keras Sequential model (CNN)
model1 = Sequential()

# Adding an input layer to which training images are passed
model1.add(InputLayer(input_shape=(28,28,1)))

# First convolutional layer with ReLU activation
model1.add(Conv2D(kernel_size=3, strides=1, filters=32, padding='same',
                 activation='relu', name='layer_conv1'))

# Second convolutional layer with ReLU activation
model1.add(Conv2D(kernel_size=3, strides=1, filters=32, padding='same',
                 activation='relu', name='layer_conv2'))

# Maxpooling layer
model1.add(MaxPooling2D(pool_size=2, strides=2))


# Third convolutional layer with ReLU activation
model1.add(Conv2D(kernel_size=3, strides=1, filters=64, padding='same',
                 activation='relu', name='layer_conv3'))

# Fourth convolutional layer with ReLU activation
model1.add(Conv2D(kernel_size=3, strides=1, filters=64, padding='same',
                 activation='relu', name='layer_conv4'))

# Maxpooling layer
model1.add(MaxPooling2D(pool_size=2, strides=2))

# Fifth convolutional layer with ReLU activation
model1.add(Conv2D(kernel_size=3, strides=1, filters=128, padding='same',
                 activation='relu', name='layer_conv5'))

# Sixth convolutional layer with ReLU activation
model1.add(Conv2D(kernel_size=3, strides=1, filters=128, padding='same',
                 activation='relu', name='layer_conv6'))

# Maxpooling layer
model1.add(MaxPooling2D(pool_size=2, strides=2))

# Seventh convolutional layer with ReLU activation
model1.add(Conv2D(kernel_size=3, strides=1, filters=256, padding='same',
                 activation='relu', name='layer_conv7'))

# Eighth convolutional layer with ReLU activation
model1.add(Conv2D(kernel_size=3, strides=1, filters=256, padding='same',
                 activation='relu', name='layer_conv8'))

# Maxpooling layer
model1.add(MaxPooling2D(pool_size=2, strides=2))

# Flatten the output of the previous layer to pass them to the Dense Layer
model1.add(Flatten())

# Dense Layer with ReLU-activation.
model1.add(Dense(1000, activation='relu'))

# Last layer with softmax activation for classification
model1.add(Dense(6, activation='softmax'))

# Nadam optimizer
optimizer1 = Nadam()

# Compile the model
model1.compile(optimizer=optimizer1,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model to training data with data augmentation
model1.fit_generator(image_gen1.flow(X_tr1, y_tr1, batch_size=64),
            validation_data=(X_eval1,y_eval1), epochs=150, verbose=1)

# Get the prediction on test data
predictions1 = model1.predict(test_imagesr)

# CNN model 2
# Train and test split
X_tr2, X_eval2, y_tr2 ,y_eval2 = train_test_split(train_imagesr, y_one_hot,
         test_size=0.1, shuffle=True,random_state=20,stratify=train_y)

#Data augmentation
image_gen2 = ImageDataGenerator(
    rescale=1,
    rotation_range=20,
    width_shift_range=.20,
    height_shift_range=.20,
    shear_range=.20,
    zoom_range=.20,
    )

# Fit the data augmentation function to the training data
image_gen2.fit(X_tr2, augment=True)

# Building a Keras Sequential model (CNN)
model2 = Sequential()

# Adding an input layer to which training images are passed
model2.add(InputLayer(input_shape=(28,28,1)))

# First convolutional layer with ReLU activation
model2.add(Conv2D(kernel_size=3, strides=1, filters=32, padding='same',
                 activation='relu', name='layer_conv1'))

# Second convolutional layer with ReLU activation
model2.add(Conv2D(kernel_size=3, strides=1, filters=32, padding='same',
                 activation='relu', name='layer_conv2'))

# Maxpooling layer
model2.add(MaxPooling2D(pool_size=2, strides=2))


# Third convolutional layer with ReLU activation
model2.add(Conv2D(kernel_size=3, strides=1, filters=64, padding='same',
                 activation='relu', name='layer_conv3'))

# Fourth convolutional layer with ReLU activation
model2.add(Conv2D(kernel_size=3, strides=1, filters=64, padding='same',
                 activation='relu', name='layer_conv4'))

# Maxpooling layer
model2.add(MaxPooling2D(pool_size=2, strides=2))

# Fifth convolutional layer with ReLU activation
model2.add(Conv2D(kernel_size=3, strides=1, filters=128, padding='same',
                 activation='relu', name='layer_conv5'))

# Sixth convolutional layer with ReLU activation
model2.add(Conv2D(kernel_size=3, strides=1, filters=128, padding='same',
                 activation='relu', name='layer_conv6'))

# Maxpooling layer
model2.add(MaxPooling2D(pool_size=2, strides=2))

# Seventh convolutional layer with ReLU activation
model2.add(Conv2D(kernel_size=3, strides=1, filters=256, padding='same',
                 activation='relu', name='layer_conv7'))

# Eighth convolutional layer with ReLU activation
model2.add(Conv2D(kernel_size=3, strides=1, filters=256, padding='same',
                 activation='relu', name='layer_conv8'))

# Maxpooling Layer
model2.add(MaxPooling2D(pool_size=2, strides=2))

# Flatten the output of the previous layer to pass them to the Dense Layer
model2.add(Flatten())

# Dense Layer with ReLU-activation.
model2.add(Dense(1000, activation='relu'))

# Last layer with softmax activation for classification
model2.add(Dense(6, activation='softmax'))

# Nadam optimizer
optimizer2 = Nadam()

# Compile the model
model2.compile(optimizer=optimizer2,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model to training data with data augmentation
model2.fit_generator(image_gen2.flow(X_tr2, y_tr2, batch_size=64),
             validation_data=(X_eval2,y_eval2), epochs=100, verbose=1)

# Get the prediction on test data
predictions2 = model2.predict(test_imagesr)

# CNN model 3
# Data augmentation
image_gen3 = ImageDataGenerator(
    rescale=1,
    rotation_range=15,
    width_shift_range=.15,
    height_shift_range=.15,
    shear_range=.15,
    zoom_range=.15,
    horizontal_flip=True,
    )

# Train and test split
X_tr3, X_eval3, y_tr3 ,y_eval3 = train_test_split(train_imagesr, y_one_hot,
                test_size=0.05,shuffle=True,random_state=70,stratify=train_y)

# Fit the data augmentation function to the training data
image_gen3.fit(X_tr3, augment=True)

# Building a Keras Sequential model (CNN)
model3 = Sequential()

# Adding an input layer to which training images are passed
model3.add(InputLayer(input_shape=(28,28,1)))

# First convolutional layer with ReLU activation
model3.add(Conv2D(kernel_size=3, strides=1, filters=32, padding='same',
                 activation='relu', name='layer_conv1'))

# Second convolutional layer with ReLU activation
model3.add(Conv2D(kernel_size=3, strides=1, filters=32, padding='same',
                 activation='relu', name='layer_conv2'))

# Maxpooling layer
model3.add(MaxPooling2D(pool_size=2, strides=2))

# Third convolutional layer with ReLU activation
model3.add(Conv2D(kernel_size=3, strides=1, filters=64, padding='same',
                 activation='relu', name='layer_conv3'))

# Fourth convolutional layer with ReLU activation
model3.add(Conv2D(kernel_size=3, strides=1, filters=64, padding='same',
                 activation='relu', name='layer_conv4'))

# Maxpooling layer
model3.add(MaxPooling2D(pool_size=2, strides=2))

# Fifth convolutional layer with ReLU activation
model3.add(Conv2D(kernel_size=3, strides=1, filters=128, padding='same',
                 activation='relu', name='layer_conv5'))

# Sixth convolutional layer with ReLU activation
model3.add(Conv2D(kernel_size=3, strides=1, filters=128, padding='same',
                 activation='relu', name='layer_conv6'))

# Maxpooling layer
model3.add(MaxPooling2D(pool_size=2, strides=2))

# Seventh convolutional layer with ReLU activation
model3.add(Conv2D(kernel_size=3, strides=1, filters=256, padding='same',
                 activation='relu', name='layer_conv7'))

# Eighth convolutional layer with ReLU activation
model3.add(Conv2D(kernel_size=3, strides=1, filters=256, padding='same',
                 activation='relu', name='layer_conv8'))

# Maxpooling layer
model3.add(MaxPooling2D(pool_size=2, strides=2))

# Flatten the output of the previous layer to pass them to the Dense Layer
model3.add(Flatten())

# Dense Layer with ReLU-activation.
model3.add(Dense(500, activation='relu'))

# Last layer with softmax activation for classification
model3.add(Dense(6, activation='softmax'))

# Nadam optimizer
optimizer3 = Nadam()

# Compile the model
model3.compile(optimizer=optimizer3,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model to training data with data augmentation
model3.fit_generator(image_gen3.flow(X_tr3, y_tr3, batch_size=128),
       validation_data=(X_eval3,y_eval3), epochs=100, verbose=1)

# Get the prediction on test data
predictions3 = model3.predict(test_imagesr)

# CNN model 4
# Data augmentation
image_gen4 = ImageDataGenerator(
    rescale=1,
    rotation_range=15,
    width_shift_range=.15,
    height_shift_range=.15,
    shear_range=.15,
    zoom_range=.15,
    horizontal_flip=True,
    )

# Fit the data augmentation function to the training data
image_gen4.fit(train_imagesr, augment=True)

# Building a Keras Sequential model (CNN)
model4 = Sequential()

# Adding an input layer to which training images are passed
model4.add(InputLayer(input_shape=(28,28,1)))

# First convolutional layer with ReLU activation
model4.add(Conv2D(kernel_size=3, strides=1, filters=32, padding='same',
                 activation='relu', name='layer_conv1'))

# Second convolutional layer with ReLU activation
model4.add(Conv2D(kernel_size=3, strides=1, filters=32, padding='same',
                 activation='relu', name='layer_conv2'))

# Maxpooling layer
model4.add(MaxPooling2D(pool_size=2, strides=2))


# Third convolutional layer with ReLU activation
model4.add(Conv2D(kernel_size=3, strides=1, filters=64, padding='same',
                 activation='relu', name='layer_conv3'))

# Fourth convolutional layer with ReLU activation
model4.add(Conv2D(kernel_size=3, strides=1, filters=64, padding='same',
                 activation='relu', name='layer_conv4'))

# Maxpooling layer
model4.add(MaxPooling2D(pool_size=2, strides=2))

# Fifth convolutional layer with ReLU activation
model4.add(Conv2D(kernel_size=3, strides=1, filters=128, padding='same',
                 activation='relu', name='layer_conv5'))

# Sixth convolutional layer with ReLU activation
model4.add(Conv2D(kernel_size=3, strides=1, filters=128, padding='same',
                 activation='relu', name='layer_conv6'))

# Maxpooling Layer
model4.add(MaxPooling2D(pool_size=2, strides=2))

# Flatten the output of the previous layer to pass them to the Dense Layer
model4.add(Flatten())

# Dense Layer with ReLU-activation.
model4.add(Dense(1000, activation='relu'))

# Last layer with softmax activation for classification
model4.add(Dense(6, activation='softmax'))

# Nadam optimizer
optimizer4 = Nadam()

# Compile the model
model4.compile(optimizer=optimizer4,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model to training data with data augmentation
model4.fit_generator(image_gen4.flow(train_imagesr, y_one_hot, 
                        batch_size=64), epochs=120, verbose=1)

# Get the prediction on test data
predictions4 = model4.predict(test_imagesr)

# Combine the predictions made by the 4 CNN models with a proper ratio
combined_prediction = 0.2*predictions1 + 0.2*predictions2 + 0.2*predictions3 + 0.4*predictions4

combined_prediction_max= np.argmax(combined_prediction, axis=1)
combined_prediction_maxs = combined_prediction_max.astype(str)

# Creating the prediction file titled 'ml_ensemble.csv'
sub =  open('ml_ensemble.csv','w+')
sub.write('Id,Category\n')
for index, prediction in enumerate(combined_prediction_maxs):
    sub.write(str(index) + ',' + prediction + '\n')
sub.close()