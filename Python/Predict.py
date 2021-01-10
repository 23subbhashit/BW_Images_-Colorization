import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
from skimage.io import imshow,imsave
from skimage.transform import resize

model = load_model('../models/weights_ImageColorization.hdf5')

img1_color=[]

img1=img_to_array(load_img("C:/DjangoProjects/BW_Images_-Colorization/google-images-download/images/national_park,dog_park_/4.jpg"))
img1 = resize(img1 ,(256,256))
img1_color.append(img1)

img1_color = np.array(img1_color, dtype=float)
img1_color = rgb2lab(1.0/255*img1_color)[:,:,:,0]
img1_color = img1_color.reshape(img1_color.shape+(1,))

output1 = model.predict(img1_color)
output1 = output1*128

result = np.zeros((256, 256, 3))
result[:,:,0] = img1_color[0][:,:,0]
result[:,:,1:] = output1[0]
imshow(lab2rgb(result))
imsave("../Results/result4.png", lab2rgb(result))