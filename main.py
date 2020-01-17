'''
Main funtion

Authors: Cindy Ku (2800612), Yung-Yu Chen (4102053683)
'''
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d

from kalmanfilter import *
from common import *
from groundtruth_generator import *
from sensor_simulator import *

'''
Given parameters
'''
v = 20 / 3600
a_y = 1
a_z = 1
a_x = 10

# radar setup
radar1 = [0, 100, 10]
radar2 = [100, 0, 10]
# range(km)
sigma_r = 0.01
# azimuth(radian), degree = 0.1
sigma_phi = math.radians(0.1)



if __name__ == "__main__":

    # Init 3D plot
    fig = plt.figure(num='Simulation of model')
    axes = fig.gca(projection='3d')
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
        
    # Generate timestep array, deltaT = 2s
    dt = 2
    timesteps = np.arange(0, a_x/v+1, dt)

    axes.plot(rx(timesteps), ry(timesteps), rz(timesteps), label = 'Ground Truth')

    ### Kalman Filter Initialization
    init_state = np.array([rx(0), ry(0), rz(0), vx(0), vy(0), vz(0)])

    F = np.array([[1, 0, 0, dt, 0 , 0], 
                  [0, 1, 0, 0, dt, 0], 
                  [0, 0, 1, 0, 0, dt], 
                  [0, 0, 0, 1, 0, 0], 
                  [0, 0, 0, 0, 1, 0], 
                  [0, 0, 0, 0, 0, 1]])

    s = 0.1
    D = s**2 * np.array([[(1/4)*(dt**4), 0, 0, (1/2)*(dt**3), 0, 0],
                         [0, (1/4)*(dt**4), 0, 0, (1/2)*(dt**3), 0],  
                         [0, 0, (1/4)*(dt**4), 0, 0, (1/2)*(dt**3)],
                         [(1/2)*(dt**3), 0, 0, dt**2, 0, 0],
                         [0, (1/2)*(dt**3), 0, 0, dt**2, 0],
                         [0, 0, (1/2)*(dt**3), 0, 0, dt**2]])

    H = np.array([[1, 0, 0, 0, 0, 0], 
                  [0, 1, 0, 0, 0, 0]]) 

    kalman = KalmanFilter(F, D, H)
    kalman.init(init_state)

    ### plot arrows
    t_step = 20
    scale_v = 100
    scale_a = 15000

    ### measurements
    z1_proj = []
    z2_proj = []
    fused_z = []

    ### track of Kalman Filter
    track = []
    
    for t_pos in range(len(timesteps)):
        t_val_start = timesteps[t_pos]
        
        vel_start = [rx(t_val_start), ry(t_val_start), rz(t_val_start)]
        vel_end = [rx(t_val_start)+vx(t_val_start)*scale_v, ry(t_val_start)+vy(t_val_start)*scale_v, rz(t_val_start)+vz(t_val_start)*scale_v] 
        
        acc_start = [rx(t_val_start), ry(t_val_start), rz(t_val_start)]
        acc_end = [rx(t_val_start)+ax(t_val_start)*scale_a, ry(t_val_start)+ay(t_val_start)*scale_a, rz(t_val_start)+az(t_val_start)*scale_a]

        ### get current measurements and transformations
        z1, z2 = compute_measurements(vel_start)
        z1_xy = cartesian_proj_transform(z1, radar1)
        z2_xy = cartesian_proj_transform(z2, radar2)

        ### fuse measurements
        R1 = compute_Rk(z1)
        R2 = compute_Rk(z2)
        
        R = np.linalg.inv( np.linalg.inv(R1) + np.linalg.inv(R2))
        z = R @ (np.linalg.inv(R1) @ z1_xy.T + np.linalg.inv(R2) @ z2_xy.T)

        ### predict and correction step
        kalman.prediction()
        kalman.update(z, R)

        ### extend list
        track.append(kalman.get_current_location())
        z1_proj.append(z1_xy)
        z2_proj.append(z2_xy)
        fused_z.append(z)
        
        if (t_pos) % 100 == 0:

            vel_vecs = list(zip(vel_start, vel_end))
            vel_arrow = Arrow3D(vel_vecs[0],vel_vecs[1],vel_vecs[2], mutation_scale=20, lw=1, arrowstyle="-|>", color="g")
            axes.add_artist(vel_arrow)

            acc_vecs = list(zip(acc_start, acc_end))
            acc_arrow = Arrow3D(acc_vecs[0],acc_vecs[1],acc_vecs[2], mutation_scale=20, lw=1, arrowstyle="-|>", color="m")
            axes.add_artist(acc_arrow)

    ### Plotting area ###

    z1_proj = np.array(z1_proj)
    z2_proj = np.array(z2_proj)
    fused_z = np.array(fused_z)
    track = np.array(track)
    zero_vec = np.zeros(len(track))

    # Retrodiction: choose timesteps in a range from 0 to 901 
    # This example starts at timestep 700 and goes back to timestep 50
    retro = kalman.retrodiction(700, 50) 

    ### Add x-y Kalman Filter in simulation plot
    axes.plot(track[:,0], track[:,1], zero_vec, color = 'r', label='Kalman Filter')
    axes.set_xlim3d(0, 10) 
    axes.set_ylim3d(-1,1)
    plt.legend()
    plt.show(block=False)

    ### Compare measurements with radars and its fusion
    plt.figure(num='Measurements of each radar (Ex. 4.1.2)')
    # compare radar 1 with ground truth
    plt.subplot(1, 3, 1)
    plt.plot(z1_proj[:,0], z1_proj[:,1], label='Radar 1')
    plt.plot(rx(timesteps), ry(timesteps), color='r', label='Ground Truth')
    plt.legend()
    # compare radar 2 with ground truth
    plt.subplot(1, 3, 2)
    plt.plot(z2_proj[:,0], z2_proj[:,1], label='Radar 2')
    plt.plot(rx(timesteps), ry(timesteps), color='r', label='Ground Truth')
    plt.legend()
    # compare fused measurements with ground truth
    plt.subplot(1, 3, 3)
    plt.plot(fused_z[:,0], fused_z[:,1], color='c', label='Fused')
    plt.plot(rx(timesteps), ry(timesteps), color='r', label='Ground Truth')
    plt.legend()
    plt.show(block=False)

    ### Compare Kalman Filter with ...
    # ... Measurements
    plt.figure(num='Comparison with Kalman Filter')
    plt.subplot(1, 2, 1)
    plt.plot(track[:,0], track[:,1], color='g', label='Kalman Filter')
    plt.plot(fused_z[:,0], fused_z[:,1], color='c', label='Measurements')
    plt.legend()
    # ... Ground Truth
    plt.subplot(1, 2, 2)
    plt.plot(track[:,0], track[:,1], color='g', label='Kalman Filter')
    plt.plot(rx(timesteps), ry(timesteps), color='r', label='Ground Truth')
    plt.legend()
    plt.show(block=False)

    ### Compare retrodiction with ... 
    plt.figure(num='Retrodiction')
    # ... Kalman Filter
    plt.subplot(1, 2, 1)
    plt.plot(track[:,0], track[:,1], color='g', label='Kalman Filter')
    plt.plot(retro[:,0], retro[:,1], color='m', label='Retrodiction')
    plt.legend()
    plt.subplot(1, 2, 2)
    # ... Ground Truth
    plt.plot(rx(timesteps), ry(timesteps), color='r', label='Ground Truth')
    plt.plot(retro[:,0], retro[:,1], color='m', label='Retrodiction')
    plt.legend()
    plt.show(block=False)

    ### Plots of Exercise 3.1.4
    ### calculate tangential vectors
    tangential_vecs = t(vx(timesteps), vy(timesteps), vz(timesteps))

    norm_v = norm(vx(timesteps), vy(timesteps), vz(timesteps))
    norm_a = norm(ax(timesteps), ay(timesteps), az(timesteps))
    
    A = make_matrix(ax(timesteps), ay(timesteps), az(timesteps))
    
    Mult = A * tangential_vecs
    product = np.sum(A * tangential_vecs, axis=0)

    plt.figure(num='Plots of Exercise 3.1.4')
    plt.plot(timesteps, norm_v, 'g', label='Norm v')
    plt.plot(timesteps, norm_a, 'r', label='Norm a')
    plt.plot(timesteps, product, 'b', label='v * Tangent')
    plt.legend()

    plt.show()  