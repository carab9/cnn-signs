#!/usr/bin/env python
# coding: utf-8

# # **Training a CNN from scratch to classify the hand-digit signs dataset**
# 
# This model will use Keras' flexible Functional API to build a ConvNet that can differentiate between 6 sign language digits - the numbers 0, 1, 2, 3, 4, and 5. The dataset consists of 1080 64x64x3 test images in 6 classes. The accuracy of the small model from scratch is around 99.17-100%. This model uses techniques such as reducing learning rate, early stopping, and data augumentation to increase the accuracy. Run time on a GPU is around 186 seconds. Loading time for data is around 0.5 seconds.
# 
# ![SIGNS%20%281%29.png](attachment:SIGNS%20%281%29.png)

# In[1]:


import math
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops
import time
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
import boto3
import botocore
import s3fs
from keras import backend
backend.set_image_data_format('channels_last')

get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(1)


# In[2]:


gpus = tf.config.list_physical_devices('GPU')
print(gpus)
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(logical_gpus)
print(tf.test.is_built_with_cuda())


# In[3]:


#s3_client = boto3.client('s3')

#train_data = s3_client.get_object(Bucket='cburgess-bucket', Key='Signs-Model/Datasets/train_signs.h5')
#test_data = s3_client.get_object(Bucket='cburgess-bucket', Key='Signs-Model/Datasets/test_signs.h5')

#print(train_data)
#print(test_data)


# In[4]:


#s3 = boto3.resource('s3')
#bucket = s3.Bucket('cburgess-bucket')
#for file in bucket.objects.all():
    #print(file.key)


# In[5]:


fs = s3fs.S3FileSystem(anon=False)
fs.ls('s3://cburgess-bucket/Signs-Model/Datasets')


# # **Processing Data**

# In[6]:


#  Load the signs data and split the data into train/test Sets
def load_signs_dataset():
    #train_dataset = h5py.File(train_data, 'r', driver='ros3')
    with fs.open('s3://cburgess-bucket/Signs-Model/Datasets/train_signs.h5') as f:
        train_dataset = h5py.File(f, 'r')
        train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
        train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
        f.close()
        
    #test_dataset = h5py.File(test_data, 'r', driver='ros3')
    with fs.open('s3://cburgess-bucket/Signs-Model/Datasets/test_signs.h5') as f:
        test_dataset = h5py.File(f, 'r')
        test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
        test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
        classes = np.array(test_dataset["list_classes"][:]) # the list of classes
        f.close()

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# In[7]:


# Loading the data
t1 = time.time()
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_signs_dataset()
print ("X_train_orig shape: " + str(X_train_orig.shape))
print ("Y_train_orig shape: " + str(Y_train_orig.shape))
print ("X_test_orig shape: " + str(X_test_orig.shape))
print ("Y_test_orig shape: " + str(Y_test_orig.shape))
print("Loading time:", time.time()-t1)


# In[8]:


# Show an example of an image from the dataset
index = 18
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))


# In[9]:


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


# In[10]:


# Normalize the data
X_train = X_train_orig/255.
X_test = X_test_orig/255.
# Reshape
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


# # **Building the model**

# In[11]:


def convolutional_model(input_shape):
  '''
  CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL ->
  CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL ->
  FLATTEN -> DENSE -> DENSE -> DENSE
  '''
  input_img = tf.keras.Input(shape=input_shape)
  # kernel_regularizer=l2(0.0001)
  Z1 = tf.keras.layers.Conv2D(64, (3,3), strides=(1,1), padding='SAME', kernel_initializer='he_uniform')(input_img)
  A1 = tf.keras.layers.ReLU()(Z1)
  P1 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='SAME')(A1)
  Z2 = tf.keras.layers.Conv2D(64, (3,3), strides=(1,1), padding='SAME', kernel_initializer='he_uniform')(P1)
  A2 = tf.keras.layers.ReLU()(Z2)
  P2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(A2)
  Z3 = tf.keras.layers.Conv2D(128, (3,3), strides=(1,1), padding='SAME', kernel_initializer='he_uniform')(P2)
  A3 = tf.keras.layers.ReLU()(Z3)
  P3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(A3)
  Z4 = tf.keras.layers.Conv2D(128, (3,3), strides=(1,1), padding='SAME', kernel_initializer='he_uniform')(P3)
  A4 = tf.keras.layers.ReLU()(Z4)
  P4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(A4)
  F = tf.keras.layers.Flatten()(P4)
  D1 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform')(F)
  D2 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform')(D1)
  outputs = tf.keras.layers.Dense(6, activation='softmax', kernel_initializer='glorot_uniform')(D2)
  conv_model = tf.keras.Model(inputs=input_img, outputs=outputs)
  return conv_model


# In[12]:


'''
def convolutional_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 3, strides=(1,1), padding='SAME', input_shape=input_shape, kernel_initializer='he_uniform'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='SAME'),
        tf.keras.layers.Conv2D(64, 3, strides=(1,1), padding='SAME', kernel_initializer='he_uniform'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='SAME'),
        tf.keras.layers.Conv2D(128, 3, strides=(1,1), padding='SAME', kernel_initializer='he_uniform'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='SAME'),
        tf.keras.layers.Conv2D(128, 3, strides=(1,1), padding='SAME', kernel_initializer='he_uniform'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='SAME'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.Dense(6, activation='softmax', kernel_initializer='glorot_uniform')
    ])
    return model
'''


# In[13]:


conv_model = convolutional_model((64, 64, 3))
conv_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
conv_model.summary()


# In[14]:


def create_callbacks():
  # Stop training when a monitored metric has stopped improving
  early_stopping = EarlyStopping(patience=5, monitor='val_loss', verbose=1,
                                 mode='min', start_from_epoch=100)

  # Reduce learning rate when a metric has stopped improving
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, min_lr=0.0001,
                                patience=5, mode='min',
                                verbose=1)

  # Save the best performing model so far to a file
  model_checkpoint = ModelCheckpoint(monitor='val_loss',
                                     filepath='./signs_best_model_s3.h5',
                                     save_best_only=True,
                                     mode='min',
                                     verbose=1)

  callbacks = [
      early_stopping,
      reduce_lr,
      model_checkpoint
  ]

  return callbacks


# # **Training the model**

# In[15]:


# Generate batches of tensor image data with real-time data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2)

datagen.fit(X_train)
t0 = time.time()
# fits the model on batches with real-time data augmentation
with tf.device('/device:GPU:0'):
    history = conv_model.fit(datagen.flow(X_train, Y_train, batch_size=16),
                             validation_data=datagen.flow(X_test, Y_test, batch_size=8),
                             steps_per_epoch=len(X_train) / 16, epochs=200,
                             callbacks=create_callbacks())
print("Training time:", time.time()-t0)


# In[16]:


# The history.history["loss"] entry is a dictionary with as many values as epochs that the
# model was trained on.
df_loss_acc = pd.DataFrame(history.history)
df_loss = df_loss_acc.loc[:, ('loss', 'val_loss')]
df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
df_acc = df_loss_acc.loc[:, ('accuracy', 'val_accuracy')]
df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
df_loss.plot(title='Model loss',figsize=(6,4)).set(xlabel='Epoch',ylabel='Loss')
df_acc.plot(title='Model Accuracy',figsize=(6,4)).set(xlabel='Epoch',ylabel='Accuracy')


# # **Loading and predicting**

# In[17]:


# Load saved best model
best_model = tf.keras.models.load_model('./signs_best_model_s3.h5')
best_model.evaluate(X_test, Y_test)

# Show the image(s) with prediction error
testgen = ImageDataGenerator()
predict_result = best_model.predict(testgen.flow(X_test, Y_test, batch_size=16, shuffle=False), steps=len(X_test)/16)
for index, prediction in enumerate(predict_result):
  predicted = np.argmax(prediction)
  label = np.squeeze(Y_test_orig[:, index])
  if predicted != label:
    print("Error:")
    print("label = " + str(label))
    print("predicted = ", predicted)
    plt.imshow(X_test_orig[index])
    plt.show()

