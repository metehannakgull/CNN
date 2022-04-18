# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 14:35:00 2022

@author: Cıvanakgul
"""
from keras.datasets import mnist
from keras import models
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#dataset
(x_train,y_train), (x_test,y_test)=mnist.load_data()

#Load model
model=models.load_model('mnist.h5')

#select a random image
inx=np.random.randint(0,10000)
rakam=x_test[inx,:,:]
plt.imshow(rakam)#imshow(rakam,cmap='gray')

#apply selected image to model
y=model.predict(rakam.reshape(1,784)/255)

pred=np.argmax(y)

print("selected image= ",inx)
print("target =",y_test[inx])
print("predicted =",pred)
