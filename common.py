'''
Helper Functions
Reference: `https://stackoverflow.com/questions/35020256/python-plotting-velocity-and-acceleration-vectors-at-certain-points`

Authors: Cindy Ku (2800612), Yung-Yu Chen (4102053683)

'''
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np

class Arrow3D(FancyArrowPatch):
    
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]    ))
        FancyArrowPatch.draw(self, renderer)

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