from skimage.color import rgb2lab, lab2rgb
import numpy as np
import glob
import cv2
from PIL import Image
import matplotlib.pyplot as plt

X=[]
Y=[]
from tqdm import tqdm
for image in tqdm(glob.glob("C:/DjangoProjects/BW_Images_-Colorization/google-images-download/images/national_park,dog_park_/*.jpg")):
    try :
        i=cv2.imread(image)
        we = Image.fromarray(i,'RGB')
        r = we.resize((256,256)) 
        lab = rgb2lab(r)
        L = lab[:,:,0]
        AB = lab[:,:,1:]
        X.append(L)
        Y.append(AB)
    except:
        pass
X=np.array(X)
Y=np.array(Y)

np.save("../Data/X_values.npy",X)
np.save("../Data/Y_values.npy",Y)