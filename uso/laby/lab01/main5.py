# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 10:40:27 2025

@author: Student
"""

import numpy as np
import matplotlib.pyplot as plt

def wartosc_wielomianu(x, w):
    #w = x4*x**4 + a[1]*x**3 + a[2]*x**2 + a[3]*x + a[4]
    wynik = 0
    #print(len(w))
    for i in range(len(w)):
        wynik += w[i]*x**i
    
    return wynik


def oblicz_min_max(w, x_array):
    max = -np.inf
    min = np.inf
    max_x = 0
    min_x = 0

    y_array = np.array([])

    for i in x_array:
        temp = np.array([wartosc_wielomianu(i, w)])
        y_array = np.append(y_array, temp)
        #print(i)
        if temp > max:
            max = temp
            max_x = i
            
        if temp < min:
            min = temp;
            min_x = i
            
    return np.array([max, min]), y_array, max_x, min_x


w = np.array([2,1,5,1,3,1])

x_array = np.linspace(-10, 10, 100)
max_x = 0
min_x = 0

min_max, y_array, max_x, min_x = oblicz_min_max(w, x_array)
#print(y_array)
print(min_max)

plt.plot(x_array, y_array, 'r', linewidth=5)
plt.plot(max_x, min_max[0], 'bo')
plt.plot(min_x, min_max[1], 'bo')

plt.title("wykres wielomianu z min max")
plt.xlabel("x")
plt.ylabel("y")
    