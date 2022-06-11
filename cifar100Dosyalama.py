# -*- coding: utf-8 -*-
"""
Cifar-100'deki görüntüleri ayrı ayrı klasörlere ayırıp tutan program

@author: metehan akgül
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from keras.datasets import cifar100
(x_train,y_train),(x_test,y_test) =cifar100.load_data()

os.mkdir('datasetCIFAR100')
os.mkdir('datasetCIFAR100\\train')
os.mkdir('datasetCIFAR100\\test')

for i in range(100):
    path=os.path.join('datasetCIFAR100\\train',str(i))
    os.mkdir(path)
    path=os.path.join('datasetCIFAR100\\test',str(i))
    os.mkdir(path)

for i in range(50000):
    path='datasetCIFAR100/train/'+str(int(y_train[i]))+'/' +str(i)+'.png'
    plt.imsave(path,x_train[i])

for i in range(10000):
    path='datasetCIFAR100/test/'+str(int(y_test[i]))+'/'+str(i)+'.png'
    plt.imsave(path,x_test[i])