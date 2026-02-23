import numpy as np
import math
from utils import temp_seed


### Training parameters ###
pos_b = 3
vel_b = 1
dis_b = 2.5
phi_b = 0.5  # np.pi/sqrt(3) , 0.5
X_MIN = np.array([-pos_b, -pos_b, -pos_b, -vel_b, -vel_b, -vel_b, -dis_b, -dis_b, -dis_b, -phi_b, -phi_b, -phi_b]).reshape(-1,1)

X_MAX = np.array([pos_b, pos_b, pos_b, vel_b, vel_b, vel_b, dis_b, dis_b, dis_b, phi_b, phi_b, phi_b]).reshape(-1,1)

g = 9.81
Tref_b = 3
dref_b = 2.5
wref_b = 0.5
UREF_MIN = np.array([-Tref_b+g, -dref_b, -dref_b, -dref_b, -wref_b, -wref_b, -wref_b]).reshape(-1,1)

UREF_MAX = np.array([Tref_b+g, dref_b, dref_b, dref_b, wref_b, wref_b, wref_b]).reshape(-1,1)

# Error bounds for training
XE_MIN = np.array([-1, -1, -1, -0.5, -0.5, -0.5, -2, -2, -2, -0.2, -0.2, -0.2]).reshape(-1,1)

XE_MAX = np.array([1, 1, 1, 0.5, 0.5, 0.5, 2, 2, 2, 0.2, 0.2, 0.2]).reshape(-1,1)

### Simulation parameters ###
r_c = 1.5
omega = 0.2
T_0 = np.sqrt(g**2 + omega**4 * r_c**2)

X_INIT_MIN = np.array([r_c, 0, 1, 0, omega*r_c, 0, 0, 0, 0, g/T_0, 0, -(omega**2*r_c)/T_0, 0, 1, 0, (omega**2*r_c)/T_0, 0, g/T_0])
X_INIT_MAX = X_INIT_MIN

# Initial error bounds for simulations
XE_INIT_MIN = np.array([-0.5, -0.5, -0.5, -0.1, -0.1, -0.1, 0, 0, 0, -0.1, -0.1, -0.1])

XE_INIT_MAX = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0, 0, 0, 0.1, 0.1, 0.1])

# Time parameters
time_bound = 2*np.pi/omega
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
        Rref_0 = xref_0[9:18].reshape(3, 3)       
        phi_e_init = xe_0[9:12]
        p_e_init = np.linalg.norm(phi_e_init)
        axis_e_init = phi_e_init/p_e_init
        Re_0 = np.cos(p_e_init)*np.eye(3) + (1 - np.cos(p_e_init))*np.outer(axis_e_init, axis_e_init) + np.sin(p_e_init)*skew(axis_e_init)
        R_0 = Rref_0 @ Re_0
        r_0 = R_0.flatten()

        # For Euclidiean states
        vref_0 = xref_0[0:9]
        ve_0 = xe_0[0:9]
        v_0 = vref_0 + ve_0

        # Initial state
        x_0 = np.concatenate([v_0, r_0]).reshape(-1,1)
        
        uref = []
        # Circular trajectory parameters
        for _t in t:
            T = np.sqrt(g**2 + omega**4 * r_c**2)
            d1 = 0
            d2 = 0
            d3 = 0
            p = (g * omega**3 * r_c * np.cos(omega*_t)) / (T * np.sqrt(g**2 + omega**4 * r_c**2 * np.sin(omega*_t)**2))
            q = (omega**3 * r_c * np.sin(omega*_t)) / np.sqrt(g**2 + omega**4 * r_c**2 * np.sin(omega*_t)**2)
            r = 0.0
            u = np.array([T, d1, d2, d3, p, q, r])
            uref.append(u)

    return x_0, xref_0, uref
