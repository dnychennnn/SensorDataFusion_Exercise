'''
Sensor Simulator

Authors: Cindy Ku (2800612), Yung-Yu Chen (4102053683)
'''

import numpy as np
import math

# radar setup
radar1 = [0, 100, 10]
radar2 = [100, 0, 10]
# range(km)
sigma_r = 0.01
# azimuth(radian), degree = 0.1
sigma_phi = math.radians(0.1)


def compute_measurements(gt):

    '''
    return z1, z2 (measurements in tuple for both radar)
    '''

    z1_r = np.sqrt((gt[0]-radar1[0])**2 + (gt[1]-radar1[1])**2 + (gt[2]-radar1[2])**2 - radar1[2]**2) +   sigma_r * np.random.normal(0,1)
    z1_phi = np.arctan2( (gt[1]-radar1[1]) , (gt[0]-radar1[0] + 0.00001) ) + sigma_phi * np.random.normal(0, 1)

    z2_r = np.sqrt((gt[0]-radar2[0])**2 + (gt[1]-radar2[1])**2 + (gt[2]-radar2[2])**2 - radar2[2]**2) +   sigma_r * np.random.normal(0,1)
    z2_phi = np.arctan2( (gt[1]-radar2[1]) , (gt[0]-radar2[0] + 0.00001) ) + sigma_phi * np.random.normal(0, 1)

    return (z1_r, z1_phi), (z2_r, z2_phi)

def cartesian_proj_transform(measurements, r_s):
    '''
    return transformation coordinates in tuple + z-axis = 0(project to x-y plane)
    '''
    (z_r, z_phi) = measurements

    z_x = z_r * np.cos(z_phi) + r_s[0]
    z_y = z_r * np.sin(z_phi) + r_s[1]

    return np.array([z_x, z_y])

def compute_Rk(z):
    D = np.array([[np.cos(z[1]), -np.sin(z[1])],
                    [np.sin(z[1]), np.cos(z[1])]])
    measurement_error = np.array([[sigma_r**2, 0], 
                            [0, (z[0]*sigma_phi)**2]])
    Rs = D @ measurement_error @ D.T

    return Rs