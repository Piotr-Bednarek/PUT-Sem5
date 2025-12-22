# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 09:53:40 2025

@author: Student
"""

import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

kp = 3
T = 2

def model(y, t):
    u = 1
    # dy = (-1/T+y + kp*u/T)
    dy = (kp*u - y)/T
    return dy

def zad2():
    system = sp.TransferFunction([kp], [T, 1])
    t, y = sp.step(system)
    plt.plot(t, y)  
    
def zad23():
    A = -1/T
    B = kp/T
    C = 1
    D = 0
    system = sp.StateSpace(A, B, C, D)
    t, y = sp.step(system)
    plt.plot(t, y)
    
def zad25():
    t = np.linspace(0, 15, 100)
    
    y = odeint(model, 0, t)

    plt.plot(t, y)
    
def zad26():    
    system = sp.TransferFunction([kp], [T, 1])
    t1, y1 = sp.step(system)
    
    A = -1/T
    B = kp/T
    C = 1
    D = 0
    system = sp.StateSpace(A, B, C, D)
    t2, y2 = sp.step(system)

    t3 = np.linspace(0, 15, 100)
    y3 = odeint(model, 0, t3)
    
    plt.figure()
    
    plt.plot(t1, y1)
    plt.plot(t2, y2)
    plt.plot(t3, y3)


R = 12
L = 1
C = 100*10**(-6)
    
def zad31():

    system = sp.TransferFunction([1], [L, R, 1/C])
    t1, y1 = sp.step(system)
    t2, y2 = sp.impulse(system)
    plt.subplot(2,1,1)
    plt.title("odp skokowa")

    plt.plot(t1, y1)
    plt.subplot(2,1,2)
    plt.title("odp impulsowa")
    plt.plot(t2, y2) 
    
    
def zad32():
    A = np.array([[0,1],[-1/(L*C),-R/L]])
    B = np.array([[0],[1/L]])
    C_array = np.array([0, 1])
    D = 0
    
    system = sp.StateSpace(A, B, C, D)
    t, y = sp.step(system)
    
    plt.plot(t, y)


def zad33():


    num_17 = [1]
    den_17 = [L, R, 1/C]
    
    # system1 = sp.StateSpace(A, B, C_array, D)

    system1 = sp.tf2ss(num_17, den_17)
  
    A = np.array([[0,1],[-1/(L*C),-R/L]])
    B = np.array([[0],[1/L]])
    C_array = np.array([0, 1])
    D = 0
    
    system2 = sp.StateSpace(A, B, C_array, D)

    system2 = sp.ss2tf(A, B, C_array, D)

    print(system1, system2)

def zad4():
    m = 1
    L = 0.5
    d = 0.1

    J = (1/3) * m * L**2

    A = np.array([[0,1],[0,-d/J]])
    B = np.array([[0],[1/J]])
    C = np.array([[1,0]])
    D = np.array([[0]])

    sys = sp.StateSpace(A, B, C, D)

    t, y = sp.step(sys, None, np.linspace(0, 10, 1000))

    plt.plot(t, y)
    plt.title('odpowiedz skokowa manipulatora')
    plt.xlabel('czas')
    plt.ylabel('kat')
    plt.grid(True)
    plt.show()

def zad42():
    m = 1
    L = 0.5
    d = 0.1

    J = (1/3) * m * L**2

    A = np.array([[0,1],[0,-d/J]])
    B = np.array([[0],[1/J]])
    C = np.array([[1,0]])
    D = np.array([[0]])

    sys = sp.StateSpace(A, B, C, D)

    t, y = sp.step(sys, None, np.linspace(0, 10, 1000))

    t = np.linspace(0, 10, 1000)
    u = np.linspace(0, 10, 1000)
    
    t1, y1, x1 = sp.lsim(sys, u, t)

    plt.plot(t1, y1, label='wyjscie y', linewidth=2)
    plt.xlabel('czas')
    plt.ylabel('amplituda')
    plt.legend()
    plt.grid(True)
    plt.show()

def zad43():
    m = 1
    L = 0.5
    d = 0.1

    J = (1/3) * m * L**2

    A = np.array([[0,1],[0,-d/J]])
    B = np.array([[0],[1/J]])
    C = np.array([[1,0]])
    D = np.array([[0]])

    sys = sp.StateSpace(A, B, C, D)

    t, y = sp.step(sys, None, np.linspace(0, 10, 1000))    


    w_bode = np.logspace(-2, 2, 500)
    w, mag, phase = sp.bode(sys, w=w_bode)

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.semilogx(w, mag)
    plt.title('bode manipulatora')
    plt.ylabel('amplituda [dB]')
    plt.grid(True, which='both', linestyle='--')

    plt.subplot(2, 1, 2)
    plt.semilogx(w, phase)
    plt.xlabel('czestotliwosc [rad/s]')
    plt.ylabel('faza [stopnie]')
    plt.grid(True, which='both', linestyle='--')

    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    zad43()