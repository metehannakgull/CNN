# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 11:17:52 2022

@author: Cıvanakgul
"""
#mnist dataset
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test)=mnist.load_data()

#Prepare dataset
x_train=x_train.reshape((60000, 28 * 28))#784
x_train=x_train.astype('float32')/255.0 #min-max normalization
x_test=x_test.reshape((10000,28*28))
x_test=x_test.astype('float32')/255.0

from tensorflow.keras.utils import to_categorical
y_train=to_categorical(y_train)#one-hot encoding
y_test=to_categorical(y_test)

from keras import models
from keras import layers, Input, Model
#from keras.models import Sequential

model=models.Sequential()
model.add(layers.Dense(32,
                       activation='relu',
                       input_shape=(784,))) # giriş sayısı
model.add(layers.Dense(32,
                       activation='relu'))
#outputs: 0,1,2,3,4,5,6,7,8,9 - multi class classifier
model.add(layers.Dense(10, #number of outputs
                       activation='softmax')) #olasılıksal çıkış verir.
#functional description -model tanımlamanın bir alternatif yazımı
inp=Input(shape=(784,))
x=layers.Dense(32,activation='relu')(inp)
x=layers.Dense(32,activation='relu')(x)
y=layers.Dense(10,activation='softmax')(x)
model=Model(inputs=inp, outputs=y)

#from keras import optimizers
model.compile(optimizer='adam',#'rmsprop', 'sgd'
              loss='categorical_crossentropy',  #'mse'
              metrics=['accuracy'])
model.summary()

#Train model - modeli eğit
model.fit(x_train,
          y_train,
          epochs=10,
          batch_size=128)

#Evaluate model - modeli değerlendir
test_loss,test_acc=model.evaluate(x_test,y_test)

print('test_acc=',test_acc)
print('test_loss=',test_loss)

#Save model - modeli kaydet
model.save('mnist.h5')
#Save weights - sadece ağırlıkları kaydet
model.save_weights('weightsmnist.h5')

