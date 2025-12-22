# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 10:13:42 2025

@author: Student
"""
import numpy as np

def wartosc_wielomianu(x, w):
    #w = x4*x**4 + a[1]*x**3 + a[2]*x**2 + a[3]*x + a[4]
    wynik = 0
    #print(len(w))
    for i in range(len(w)):
        wynik += w[i]*x**i
    
    return wynik


def oblicz_min_max(w, gr1, gr2, step):
    max = -np.inf
    min = np.inf
    max_x = 0
    min_x = 0

    array = np.linspace(gr1, gr2, step)

    for i in array:
        temp = wartosc_wielomianu(i, w)
        #print(i)
        if temp > max:
            max = temp
            max_x = i
            
        if temp < min:
            min = temp;
            min_x = i
            
    
    return np.array([max, min])


w = np.array([1,3,3])

#print(wartosc_wielomianu(10, w))

min_max = oblicz_min_max(w, 50, 100, 30000)
    