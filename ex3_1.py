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
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]    ))
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
lambda_r = 0.01
# azimuth(degree)
lambda_phi = 0.1 


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

    z1_r = np.sqrt((gt[0]-radar1[0])**2 + (gt[1]-radar1[1])**2 + (gt[2]-radar1[2])**2 - radar1[2]**2) +   lambda_r * np.random.normal(mu, sigma)
    z1_phi = np.arctan( (gt[1]-radar1[1]) / (gt[0]-radar1[0]) ) + lambda_phi * np.random.normal(mu, sigma)

    z2_r = np.sqrt((gt[0]-radar2[0])**2 + (gt[1]-radar2[1])**2 + (gt[2]-radar2[2])**2 - radar2[2]**2) +   lambda_r * np.random.normal(mu, sigma)
    z2_phi = np.arctan( (gt[1]-radar2[1]) / (gt[0]-radar2[0]) ) + lambda_phi * np.random.normal(mu, sigma)


    return (z1_r, z1_phi), (z2_r, z2_phi)

def cartesian_proj_transform(measurements, r_s):
    '''
    return transformation coordinates in tuple + z-axis = 0(project to x-y plane)
    '''
    (z_r, z_phi) = measurements

    z_x = z_r * np.cos(z_phi) + r_s[0]
    z_y = z_r * np.sin(z_phi) + r_s[1]

    return (z_x, z_y, 0)


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
        self.convariance = None

    def init(self, init_state):
        self.state = init_state
        self.convariance = np.eye(init_state.shape[0]) * 0.01

    def track(self, xt):
        
        pred_state =  self.Lambda @ self.state.T
        pred_covariance = self.sigma_p + self.Lambda @ self.convariance @ self.Lambda.T
        kalman_gain = pred_covariance @ self.Phi.T @ np.linalg.inv(self.sigma_m + self.Phi @ (pred_covariance @ self.Phi.T))

        ##debug
        update_state = pred_state + kalman_gain @ (xt - self.Phi @ pred_state)
        update_covariance = (np.identity(kalman_gain.shape[0]) - kalman_gain @ self.Phi) @ pred_covariance
        self.state = update_state
        self.convariance = update_covariance
        pass

    def get_current_location(self):
        return self.Phi @ self.state


if __name__ == "__main__":


    fig = plt.figure(num='Task3')
    axes = fig.gca(projection='3d')
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
        
    timesteps = np.linspace(0, a_x/v, 1000)

    axes.plot(rx(timesteps), ry(timesteps), rz(timesteps))

    # Kalman Filter Initialization
    init_state = np.array([rx(0), ry(0), vx(0), vy(0)])

    print(init_state)

    dt = 1
    Lambda = np.array([[1, 0, dt , 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0,1]])

    sp = 0.01
    sigma_p = np.array([[sp, 0, 0, 0],
                        [0, sp, 0, 0],
                        [0, 0, sp * 4, 0],
                        [0, 0, 0, sp * 4]])

    Phi = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]) 

    sm = 0.05
    sigma_m = np.array([[sm, 0], [0, sm]])

    tracker = KalmanFilter(Lambda, sigma_p, Phi, sigma_m)
    tracker.init(init_state)

    

    ### plot arrows
    t_step = 10
    scale_v = 0.01
    scale_a = 0.001

    ### track
    track = []
    
    for t_pos in range(0, len(timesteps)-1, t_step):
        t_val_start = timesteps[t_pos]
        
        vel_start = [rx(t_val_start), ry(t_val_start), rz(t_val_start)]
        vel_end = [rx(t_val_start)+vx(t_val_start)*scale_v, ry(t_val_start)+vy(t_val_start)*scale_v, rz(t_val_start)+vz(t_val_start)*scale_v] 
        
        acc_start = [rx(t_val_start), ry(t_val_start), rz(t_val_start)]
        acc_end = [rx(t_val_start)+ax(t_val_start)*scale_a, ry(t_val_start)+ay(t_val_start)*scale_a, rz(t_val_start)+az(t_val_start)*scale_a]
       
        if (t_pos) % 100 == 0:

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

            ### plot radar measurements 
            axes.scatter(*z1_xy, c='b')
            axes.scatter(*z2_xy, c='g')

            ### Kalman Filter Tracker

        
        
            tracker.track(np.asarray(z1_xy)[:2]) 
            estimation = tracker.get_current_location()
            track.append(estimation)
            p = Circle((estimation[0], estimation[1]), .2, color='red', fill=False)
            axes.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=0)


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




