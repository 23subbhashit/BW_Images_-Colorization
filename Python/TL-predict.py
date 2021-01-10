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
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16

model = load_model('../models/weights_ImageColorization_VGG19.hdf5')
newmodel =load_model("../models/weights_newmodel_VGG19.hdf5")

print(model.summary())

test = img_to_array(load_img("C:/DjangoProjects/BW_Images_-Colorization/google-images-download/images/national_park,dog_park_/5.jpg"))
test = resize(test, (256,256), anti_aliasing=True)
test*= 1.0/255
lab = rgb2lab(test)
l = lab[:,:,0]
L = gray2rgb(l)
L = L.reshape((1,256,256,3))
#print(L.shape)
vggpred = newmodel.predict(L)
ab = model.predict(vggpred)
#print(ab.shape)
ab = ab*128
cur = np.zeros((256, 256, 3))
cur[:,:,0] = l
cur[:,:,1:] = ab
imsave("../Results/result5_VGG16.png", lab2rgb(cur))