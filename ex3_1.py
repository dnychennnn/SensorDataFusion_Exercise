import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import FancyArrowPatch, Circle
import mpl_toolkits.mplot3d.art3d as art3d



# from `https://stackoverflow.com/questions/35020256/python-plotting-velocity-and-acceleration-vectors-at-certain-points`
class Arrow3D(FancyArrowPatch):
    
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

origin = np.zeros((3, 1))
v = 20
a_y = 1
a_z = 1
a_x = 10

# radar setup
radar1 = [0, 100, 10]
radar2 = [100, 0, 10]
# range(km)
sigma_r = 0.01
# azimuth(degree)
sigma_phi = 0.1 


# Helper functions
def array(ls):
    return np.asarray(ls)

def make_matrix(x, y, z):
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    M = np.array([x, y, z])
    return M

def norm(x, y, z):
    M = make_matrix(x, y, z)
    norm = np.linalg.norm(M, axis=0)
    return norm

'''
Ground Truth Generator
'''
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


'''
Sensor Simulator
'''


def compute_measurements(gt):

    '''
    return z1, z2 (measurements in tuple for both radar)
    '''

    mu = 0
    sigma = .1

    z1_r = np.sqrt((gt[0]-radar1[0])**2 + (gt[1]-radar1[1])**2 + (gt[2]-radar1[2])**2 - radar1[2]**2) +   sigma_r * np.random.normal(mu, sigma)
    z1_phi = np.arctan( (gt[1]-radar1[1]) / (gt[0]-radar1[0] + 0.00001) ) + sigma_phi * np.random.normal(mu, sigma)

    z2_r = np.sqrt((gt[0]-radar2[0])**2 + (gt[1]-radar2[1])**2 + (gt[2]-radar2[2])**2 - radar2[2]**2) +   sigma_r * np.random.normal(mu, sigma)
    z2_phi = np.arctan( (gt[1]-radar2[1]) / (gt[0]-radar2[0] + 0.00001) ) + sigma_phi * np.random.normal(mu, sigma)


    return [z1_r, z1_phi], [z2_r, z2_phi]

def cartesian_proj_transform(measurements, r_s):
    '''
    return transformation coordinates in tuple + z-axis = 0(project to x-y plane)
    '''
    (z_r, z_phi) = measurements

    z_x = z_r * np.cos(z_phi) + r_s[0]
    z_y = z_r * np.sin(z_phi) + r_s[1]

    return [z_x, z_y, 0]


'''
Kalman Filter
'''

class KalmanFilter(object):
    def __init__(self, Lambda, sigma_p, Phi, sigma_m):
        self.Lambda = Lambda
        self.sigma_p = sigma_p
        self.Phi = Phi
        self.sigma_m = sigma_m
        self.state = None
        self.covariance = None

    def init(self, init_state):
        self.state = init_state
        self.covariance = np.eye(init_state.shape[0]) * 0.01

    def track(self, xt):
        
        pred_state =  self.Lambda @ self.state.T
        pred_covariance = self.sigma_p + self.Lambda @ self.covariance @ self.Lambda.T
        kalman_gain = pred_covariance @ self.Phi.T @ np.linalg.inv(self.sigma_m + self.Phi @ (pred_covariance @ self.Phi.T))

        ##debug
        update_state = pred_state + kalman_gain @ (xt - self.Phi @ pred_state)
        update_covariance = (np.identity(kalman_gain.shape[0]) - kalman_gain @ self.Phi) @ pred_covariance
        self.state = update_state
        self.covariance = update_covariance
        pass
    
    # what exactly is input state & cov? Maybe self.state, self.covariance
    # need history of past states and covariances
    # def retrodiction(self, timestep, state, cov):
    #     past_state = states[timestep]
    #     pred_past_state = self.Lambda @ pred_state.T
    #     past_covariance = cov[timestep]
    #     pred_past_covariance = self.sigma_p + self.Lambda @ past_covariance @ self.Lambda.T

    #     W = past_covariance @ self.Lambda @ np.linalg.inv(pred_past_covariance)
    #     retro_state = past_state + W @ (state - pred_past_state)
    #     retro_cov = past_covariance + W  @ (cov - pred_past_covariance) @ W.T
        
    #     return retro_state, retro_cov

    def get_current_location(self):
        return self.Phi @ self.state


if __name__ == "__main__":


    fig = plt.figure(num='Task3')
    axes = fig.gca(projection='3d')
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
        
    # make delta T = 2s
    timesteps = np.linspace(0, a_x/v, 900) 

    axes.plot(rx(timesteps), ry(timesteps), rz(timesteps))

    # Kalman Filter Initialization
    
    init_state = np.array([rx(0), ry(0), rz(0), vx(0), vy(0), vz(0)])

    
    
    dt = 1 / 1800
    Lambda = np.array([[1, 0, 0, dt, 0 , 0], 
                        [0, 1, 0, 0, dt, 0], 
                        [0, 0, 1, 0, 0, dt], 
                        [0, 0, 0, 1, 0, 0], 
                        [0, 0, 0, 0, 1, 0], 
                        [0, 0, 0, 0, 0, 1]])
    
 
    sigma_k = 100 
    sigma_p = sigma_k**2 * np.array([[1/4 * (dt**4), 0, 0, 1/2*(dt**3), 0, 0],
                                    [0, 1/4 * (dt**4), 0, 0, 1/2*(dt**3), 0],  
                                    [0, 0, 1/4 * (dt**4), 0, 0, 1/2*(dt**3)],
                                    [1/2*(dt**3), 0, 0, dt**2, 0, 0],
                                    [0, 1/2*(dt**3), 0, 0, dt**2, 0],
                                    [0, 0, 1/2*(dt**3), 0, 0, dt**2]])
                    


    Phi = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]) 

    sigma_m = np.array([[sigma_r**2, 0], [0, sigma_phi**2]])

    # Init Kalman filter
    tracker1 = KalmanFilter(Lambda, sigma_p, Phi, sigma_m)
    tracker1.init(init_state)
    
    tracker2 = KalmanFilter(Lambda, sigma_p, Phi, sigma_m)
    tracker2.init(init_state)

    ### plot arrows
    t_step = 10
    scale_v = 0.01
    scale_a = 0.001

    ### track
    track = []
    alpha = 0
    for t_pos in range(0, len(timesteps)-1, t_step):
        t_val_start = timesteps[t_pos]
        
        vel_start = [rx(t_val_start), ry(t_val_start), rz(t_val_start)]
        vel_end = [rx(t_val_start)+vx(t_val_start)*scale_v, ry(t_val_start)+vy(t_val_start)*scale_v, rz(t_val_start)+vz(t_val_start)*scale_v] 
        
        acc_start = [rx(t_val_start), ry(t_val_start), rz(t_val_start)]
        acc_end = [rx(t_val_start)+ax(t_val_start)*scale_a, ry(t_val_start)+ay(t_val_start)*scale_a, rz(t_val_start)+az(t_val_start)*scale_a]
       
        if (t_pos) % 10 == 0:

            if t_pos % 100 ==0 :        
                vel_vecs = list(zip(vel_start, vel_end))
                vel_arrow = Arrow3D(vel_vecs[0],vel_vecs[1],vel_vecs[2], mutation_scale=20, lw=1, arrowstyle="-|>", color="g")
                axes.add_artist(vel_arrow)

                acc_vecs = list(zip(acc_start, acc_end))
                acc_arrow = Arrow3D(acc_vecs[0],acc_vecs[1],acc_vecs[2], mutation_scale=20, lw=1, arrowstyle="-|>", color="m")
                axes.add_artist(acc_arrow)


            ### get current measurements and transformation
            z1, z2 = compute_measurements(vel_start)
            z1_xy = cartesian_proj_transform(z1, radar1)
            z2_xy = cartesian_proj_transform(z2, radar2)
            z2_xy[0] = z2_xy[0]-190     
            

            if t_pos % 100 == 0:
                ### plot radar measurements 
                axes.scatter(*z1_xy, c='b')
                axes.scatter(*z2_xy, c='g')


            ### Kalman Filter Tracker1
            tracker1.track(np.asarray(z1_xy)[:2].T) 
            estimation1 = tracker1.get_current_location()

            tracker2.track(np.asarray(z2_xy)[:2].T)
            estimation2 = tracker2.get_current_location()

            track.append(estimation1)
            if t_pos % 50 ==0 :        
                alpha+=0.05
                p1 = Circle((estimation1[0], estimation1[1]), .2, color='red', fill=False, alpha=alpha)
                axes.add_patch(p1)
                art3d.pathpatch_2d_to_3d(p1, z=0)   

                p2 = Circle((estimation2[0], estimation2[1]), .2, color='green', fill=False, alpha=alpha)
                axes.add_patch(p2)
                art3d.pathpatch_2d_to_3d(p2, z=0) 



    axes.set_xlim3d(0, 10)
    axes.set_ylim3d(-1,1)
    # axes.set_zlim3d(0,1)

    plt.show(block=False)


    ## calculate tangential vectors
    tangential_vecs = t(vx(timesteps), vy(timesteps), vz(timesteps))


    ## Task 3.1.4
    norm_v = norm(vx(timesteps), vy(timesteps), vz(timesteps))
    norm_a = norm(ax(timesteps), ay(timesteps), az(timesteps))
    
    A = make_matrix(ax(timesteps), ay(timesteps), az(timesteps))

    
    Mult = A * tangential_vecs
    product = np.sum(A * tangential_vecs, axis=0)

    plt.figure(num='Task 4')
    plt.plot(timesteps, norm_v, 'g', label='norm v')
    plt.plot(timesteps, norm_a, 'r', label='norm a')
    plt.plot(timesteps, product, 'b', label='v * tangent')
    plt.legend()

    plt.show()  




