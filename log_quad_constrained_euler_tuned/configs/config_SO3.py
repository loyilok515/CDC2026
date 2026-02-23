import numpy as np
import math
from utils import temp_seed

### Training parameters ###
phi_b = 0.5  # np.pi/sqrt(3) , 0.5
X_MIN = np.array([-phi_b, -phi_b, -phi_b, -1, -1, -1]).reshape(-1,1)

X_MAX = np.array([phi_b, phi_b, phi_b, 1, 1, 1]).reshape(-1,1)

UREF_MIN = np.array([-0.5, -0.5, -0.5]).reshape(-1,1)

UREF_MAX = np.array([0.5, 0.5, 0.5]).reshape(-1,1)

# Error bounds for training
XE_MIN = np.array([-0.2, -0.2, -0.2, -0.5, -0.5, -0.5]).reshape(-1,1)

XE_MAX = np.array([0.2, 0.2, 0.2, 0.5, 0.5, 0.5]).reshape(-1,1)

### Simulation parameters ###

X_INIT_MIN = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0.1, -0.1, 0.8])
X_INIT_MAX = X_INIT_MIN


# Initial error bounds for simulations
XE_INIT_MIN = np.array([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1])

XE_INIT_MAX = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

# Time parameters
time_bound = 12  # Circular path period = 63s, Figure-8 path period = 63s
time_step = 0.01
t = np.arange(0, time_bound, time_step)

def skew(v):
    """
    Returns the skew-symmetric matrix of a 3D vector.
    """
    v = v.flatten()
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def system_reset(seed):
    SEED_MAX = 10000000    
    with temp_seed(int(seed * SEED_MAX)):
        xref_0 = X_INIT_MIN # + np.random.rand(len(X_INIT_MIN)) * (X_INIT_MAX - X_INIT_MIN)
        xe_0 = XE_INIT_MIN + np.random.rand(len(XE_INIT_MIN)) * (XE_INIT_MAX - XE_INIT_MIN)
        
        # For SO3
        Rref_0 = xref_0[0:9].reshape(3, 3)       
        phi_e_init = xe_0[0:3]
        p_e_init = np.linalg.norm(phi_e_init)
        axis_e_init = phi_e_init/p_e_init
        Re_0 = np.cos(p_e_init)*np.eye(3) + (1 - np.cos(p_e_init))*np.outer(axis_e_init, axis_e_init) + np.sin(p_e_init)*skew(axis_e_init)
        R_0 = Rref_0 @ Re_0
        r_0 = R_0.flatten()

        # For Euclidiean states
        vref_0 = xref_0[9:12]
        ve_0 = xe_0[3:6]
        v_0 = vref_0 + ve_0

        # Initial state
        x_0 = np.concatenate([r_0, v_0]).reshape(-1,1)
        
        uref = []
        for _t in t:
            u = np.array([0.0, 0.0, 0.0])
            uref.append(u)

    return x_0, xref_0, uref
