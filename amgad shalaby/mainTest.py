import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BrainTumor10EpochsCategorical.h5')

imge = cv2.imread('C:\\Users\\hpc\\Downloads\\weeeeb\\archive\\pred\\pred5.jpg')
img = Image.fromarray(imge)
img = img.resize((64,64))
img = np.array(img)
print(img)

from keras.backend import argmax
input_img = np.expand_dims(img,axis=0)
predictions = model.predict(input_img)
classes_x=argmax(predictions,axis=1)
print(classes_x)


if classes_x == 0:
  print('the MRI is NOT having a brain Tumor')
else:
  print('the MRI is having a brain Tumor')
