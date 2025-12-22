# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 10:09:07 2025

@author: Student
"""

import numpy as np

def wartosc_wielomianu(x):
    a = np.array([1,1,-129,171,1620])
    w = a[0]*x**4 + a[1]*x**3 + a[2]*x**2 + a[3]*x + a[4]
    return w

max = -np.inf
min = np.inf
max_x = 0
min_x = 0

array = np.linspace(-46, 14, 100000)

for i in array:
    temp = wartosc_wielomianu(i)
    #print(i)
    if temp > max:
        max = temp
        max_x = i
        
    if temp < min:
        min = temp;
        min_x = i
        


print("max x:", max_x)
print("min x:", min_x)

print("max:", max)

print("min:", min)
    
    