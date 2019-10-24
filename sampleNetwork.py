#This code retrains the top convolusional block of vgg16 and uses a custom classifier on top
#to classify male/female faces

#This version uses all male available inputs by using sets of training data
#Achieved loss of 0.158


import numpy as np
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense

#This was necessary on MacOS, comment these two lines if you're not on MacOS
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


#Path of directory holding training data, separated into subdirectories by label
train_path = "train/"
#Path of directory holding validation data categorized as train data is
validation_path = "validation/"
#Path of directory holding test data, all in one subdirectory
test_path = "data/test/"
#Determine uniform input size of images
img_width, img_height = 224,224
#Set batch size to 10
batch_size = 10
#Total number of samples
total_samples = 28360
#Designate number of train and validation samples
train_samples = int(((total_samples*0.8)//batch_size)*batch_size)
# train_samples_per_set = int(train_samples/3)
validation_samples = int(((total_samples*0.2)//batch_size)*batch_size)
#Designate amount of test samples
# test_samples = 7090
#Number of epochs for training
epochs = 1

#Create train generator that performs sheer to create a more rigorous model
#Also rescales images to work for VGG16(rbg values from 0-1 instead of 0-255)
#See https://keras.io/preprocessing/image/ to see and experiment with other transformations
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0)
#Set validation generator only to rescale images
validation_datagen = ImageDataGenerator(rescale = 1./255)
#Create an empty sequential model
model = Sequential()
#Create empty, headless vgg16 model to put into sequential model
#Applications has other (perhaps better) models. Check keras documentation
vgg_model = applications.VGG16(include_top = False, weights = 'imagenet', input_shape=(img_width,img_height,3))
#Freeze all layers below classifier
for layer in vgg_model.layers:
  layer.trainable = False
#Make headless vgg16 base of sequential model
model.add(vgg_model)
#Add top of model(classifier), consisting of Flatten -> Dense(relu with dropout, 0.5) -> softmax(one node, male or not)
model.add(Flatten(input_shape=(vgg_model.output_shape[1:])))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0))
model.add(Dense(1, activation='sigmoid'))

# print("Length of top model: ", len(model.layers))
"""
Compile model with binary crossentropy as loss function, use stochastic gradient descent as optimizer
learning rate of 0.0001, with momentum of 0. Measure using accuracy(and loss)
Change the momentum to some value
"""
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0),
              metrics=['accuracy'])

#Populate validation generator from training samples held in train_path
validation_generator = validation_datagen.flow_from_directory(
        validation_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

#Populate training generator from training samples held in train_path
train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

checkpoint = callbacks.ModelCheckpoint('final_conv_weights_check.h5', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)

model.fit_generator(
        train_generator,
        steps_per_epoch=train_samples // batch_size,#Each epoch should iterate over entire training set
        epochs=epochs,
        callbacks = [checkpoint],
        validation_data=validation_generator,
        validation_steps=validation_samples // batch_size)#Validation should use entire validation set


#Save weights of trained model
model.save_weights("final_conv_weights.h5")
