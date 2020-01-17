'''
Ground Truth Generator

Authors: Cindy Ku (2800612), Yung-Yu Chen (4102053683)
'''
import numpy as np
from common import *


# Given parameters
v = 20 / 3600
a_y = 1
a_z = 1
a_x = 10

# Position Equation
def rx(t):
    return v*t
def ry(t):
    return a_y * np.sin((4*np.pi*v)/a_x *t)
def rz(t):
    return a_z * np.sin((np.pi*v)/a_x * t)

# Velocity Vectors
def vx(t):
    return v*t/(t+0.000001)
def vy(t):
    return (4*np.pi*v/a_x) *a_y*np.cos((4*np.pi*v)/a_x *t)               
def vz(t):
    return (np.pi*v/a_x) *a_z*np.cos((np.pi*v)/a_x *t)

# Acceleration Vectors
def ax(t):
    return 0*t/(t+0.0000001)
def ay(t):
    return -(4*np.pi*v/a_x)**2 *a_y*np.sin((4*np.pi*v)/a_x *t)
def az(t):
    return -(np.pi*v/a_x)**2 *a_z*np.sin((np.pi*v)/a_x *t)

# Tangential Unit Vectors
def t(vx, vy, vz):
   
    V = make_matrix(vx, vy, vz)
    norm = np.linalg.norm(V, axis=0)

    t = V / norm
    return t 