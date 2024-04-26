import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BrainTumor10Epochscategorical.h5')

image=cv2.imread('J:\\final_project\\pred\\pred23.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

print(img)

input_img=np.expand_dims(img,axis=0)

#result=model.predict_classes(img)
result = model.predict(input_img)

#result_final=np.argmax(result)
#print(result_final)
output=np.argmax(result)

print(output)