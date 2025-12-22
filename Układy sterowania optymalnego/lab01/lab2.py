# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 09:48:14 2025

@author: Student
"""


import numpy as np
from numpy.linalg import matrix_rank


x = 3**12 - 5

print(x)

a = np.array([2, 0.5])
b = np.array([[1, 4], [-1, 3]])
c = np.array([[-1], [-3]])

x1 = np.multiply(a, b)
x1 = np.multiply(x1, c)
print(x1)

x2 = np.array([[1,-2,0],[-2,4,0],[2,-1,7]])

matrix_rank(x2)

q = np.array([[-1],[2]])
w = np.array([[1,2],[-1,0]])

x3 = np.linalg.solve(w, q)