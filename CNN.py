# -*- coding: utf-8 -*-
"""

Training the convnet on MNIST images
Bu kod önemli sınavda çıkar 6. hafta sanal sınıf
"""
from keras import layers
from keras import models

model = models.Sequential()
                                   
#dolgulama yapmak için padding=same seçilir
#32 değeri 32 farklı kernel matrisi ifade eder.
#Yani 1. katmanımızdaki çıkış sayısı da 32 olmuş oldu.
#Kernel matrisinin boyutu da (3,3) yani 3x3 boyutundadır.
model.add(layers.Conv2D(32,(3,3), 
                        #strides=(2,2)#2x2 boyutunda stride
                        use_bias=False, #True
                        padding='same', #padding belirtmezsek default olarak valid alır.
                        activation='relu',input_shape=(28,28,1)))

model.add(layers.MaxPooling2D((2,2))) #maxpooling deafult olarak 2x2 boyutundadır.

model.add(layers.Conv2D(64,  
                        (3,3),
                        padding='same',
                        activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,
                        (3,3),
                        padding='same',
                        activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.summary() 
#Çıkan parametrelerin nasıl hesaplandığını bilmeliyiz sınavda çıkar.