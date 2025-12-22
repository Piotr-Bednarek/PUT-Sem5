# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 10:02:13 2025

@author: Student
"""
import numpy as np

a = np.array([1,1,-129,171,1620])

x = -46

w = a[0]*x**4 + a[1]*x**3 + a[2]*x**2 + a[3]*x + a[4]

print(w)

x = 14

w = a[0]*x**4 + a[1]*x**3 + a[2]*x**2 + a[3]*x + a[4]

print(w)