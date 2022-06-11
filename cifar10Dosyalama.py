# -*- coding: utf-8 -*-
"""
Cifar-10daki görüntüleri ayrı ayrı klasörlere ayırıp tutan program

@author: Cıvanakgul
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.datasets import cifar10
(x_train,y_train),(x_test,y_test) =cifar10.load_data()

os.mkdir('datasetCIFAR10')
os.mkdir('datasetCIFAR10\\train')
os.mkdir('datasetCIFAR10\\test')

for i in range(10):
    path=os.path.join('datasetCIFAR10\\train',str(i))
    os.mkdir(path)
    path=os.path.join('datasetCIFAR10\\test',str(i))
    os.mkdir(path)

for i in range(50000):
    path='datasetCIFAR10/train/'+str(int(y_train[i]))+'/' +str(i)+'.png'
    plt.imsave(path,x_train[i])

for i in range(10000):
    path='datasetCIFAR10/test/'+str(int(y_test[i]))+'/'+str(i)+'.png'
    plt.imsave(path,x_test[i])







































