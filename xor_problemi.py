# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 19:19:39 2022

@author: Cıvanakgul
"""
#XOR PROBLEMİ

import numpy as np

X=np.array([[0,0],[0,1],[1,0],[1,1]])#giriş
Y=np.array([0,1,1,0]) #çıkış

#model
#from keras.models import Sequential
#model=Sequential()

from keras import models
from keras import layers
model=models.Sequential()
model.add(layers.Dense(units=5,
                       activation='relu',
                       input_shape=(2,))) 
#layer 1'den sonraki katmanlarda input-shape ihtiyaç yoktur.
#model.add(layers.Dense(units=5,
#                       activation='relu'))
model.add(layers.Dense(units=1,
                       activation='sigmoid'))
model.summary()

#compile
import tensorflow as tf
from keras import losses
#sgd -stochastic gradient
model.compile( optimizer=tf.keras.optimizers.SGD(lr=1e-3),
              loss=losses.binary_crossentropy,
              metrics=['acc','mae'])#'binary_crossentropy'
#eğitim - training
h=model.fit(X,Y,
          batch_size=1,
          epochs=10)
#tahmin
ypred=model.predict(X)
print('yp=',ypred)
y_round=np.round(ypred)
print('yt=',y_round)

import matplotlib.pyplot as plt
t_acc=h.history['acc']
t_loss=h.history['loss']

#accuracy grafiği
plt.figure(0)
plt.plot(t_acc)
plt.xlabel('epochs')
plt.ylabel('accuracy')

#loss grafiği
plt.figure(1)
plt.plot(t_loss)
plt.xlabel('epochs')
plt.ylabel('accuracy')



















