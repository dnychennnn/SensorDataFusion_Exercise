'''
Kalman Filter
'''

import numpy as np

class KalmanFilter(object):
    def __init__(self, F, D, H):
        self.F = F
        self.D = D
        self.H = H
        self.R = None
        self.x = None
        self.P = None
        self.states = []
        self.cov = []

    def init(self, init_state):
        self.x = init_state
        self.P = np.eye(init_state.shape[0]) * 0.1
        self.states.append(self.x)
        self.cov.append(self.P)

    def prediction(self):
        self.x = self.F @ self.x.T
        self.P = self.F @  self.P @ self.F.T + self.D
    
    def update(self, z_t, R):
        self.R = R
        S = self.H @ self.P @ self.H.T + self.R
        nu = z_t - self.H @ self.x.T
        W = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x.T + W @ nu
        self.P = self.P - W @ S @ W.T 

        self.states.append(self.x)
        self.cov.append(self.P)
        pass

    def retrodiction(self, cur_timestep, past_timestep):
        cur_state = self.states[cur_timestep]
        cur_cov = self.cov[cur_timestep]
        retro_states = [self.H @ cur_state]
        t = cur_timestep

        while t != past_timestep:
            x_ll = self.states[t-1]
            P_ll = self.cov[t-1]
            pred_xll = self.F @ x_ll.T
            pred_Pll = self.F @  P_ll @ self.F.T + self.D

            W = P_ll @ self.F.T @ np.linalg.inv(pred_Pll)
            cur_state = x_ll + W @ (cur_state - pred_xll).T
            cur_cov = P_ll + W @ (cur_cov - pred_Pll) @ W.T

            retro_states.append(self.H @ cur_state)
            t -= 1

        return np.array(retro_states)

    def get_current_location(self):
        return self.H @ self.x