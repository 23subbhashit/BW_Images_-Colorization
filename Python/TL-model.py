from tensorflow.keras.layers import Conv2D, UpSampling2D, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, gray2rgb
from tensorflow.keras.callbacks import ModelCheckpoint
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import tensorflow as tf
import keras
import os
from tensorflow.keras.applications.vgg16 import VGG16

X = np.load("../Data/X_values.npy")
Y = np.load("../Data/Y_values.npy")
Y=Y/128

print(X.shape,Y.shape)
X = X.reshape(X.shape+(1,)) #dimensions to be the same for X and Y
print(X.shape,Y.shape)

vggmodel = VGG16()
newmodel = Sequential() 

for i, layer in enumerate(vggmodel.layers):
    if i<19:          
      newmodel.add(layer)
newmodel.summary()
for layer in newmodel.layers:
  layer.trainable=False

newmodel.save("../models/weights_newmodel_VGG19.hdf5")

vggfeatures = []
for i, sample in enumerate(X):
  sample = gray2rgb(sample)
  sample = sample.reshape((1,256,256,3))
  prediction = newmodel.predict(sample)
  prediction = prediction.reshape((8,8,512))
  vggfeatures.append(prediction)
vggfeatures = np.array(vggfeatures)
print(vggfeatures.shape)


model = Sequential()

model.add(Conv2D(256, (3,3), activation='relu', padding='same', input_shape=(8,8,512)))
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
model.summary()


model.compile(optimizer='Adam', loss='mse' , metrics=['accuracy'])
model.summary()


filepath="../models/weights_ImageColorization_VGG19.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
history=model.fit(vggfeatures,Y,batch_size=16,epochs=10,verbose=1,validation_split=0.1,callbacks=[checkpoint])

