# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 15:54:48 2022

@author: Cıvanakgul
"""
import numpy as np

t1=np.array([1.0,0.0,0.0])#target
p1=np.array([0.7,0.1,0.2])#predicted

t2=np.array([0.0,1.0,0.0])#target
p2=np.array([0.5,0.2,0.3])#predicted

t3=np.array([0.0,0.0,1.0])#target
p3=np.array([0.1,0.2,0.6])#predicted

eps=0#1e-18
#categorical cross entropy
ce=-(np.matmul(t1,np.log(p1+eps))+np.matmul(t2,np.log(p2+eps))+np.matmul(t3,np.log(p3+eps)))

print("categorical ce=",ce)