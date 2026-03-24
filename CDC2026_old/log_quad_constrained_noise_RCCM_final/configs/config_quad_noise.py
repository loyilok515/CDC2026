import numpy as np
import math
from utils import temp_seed, euler_xyz_to_rotmat


### Training parameters ###
# Training bounds for x
pos_b = 5  # Position
vel_b = 2  # Velocity
roll_b = 1  # Roll (rad)
pitch_b = 1  # Pitch (rad)
yaw_b = np.pi + 0.1  # Yaw (rad)(Train for all possible yaw angles)

X_MIN = np.array([-pos_b, -pos_b, -pos_b, -vel_b, -vel_b, -vel_b, -roll_b, -pitch_b, -yaw_b]).reshape(-1,1)
X_MAX = np.array([pos_b, pos_b, pos_b, vel_b, vel_b, vel_b, roll_b, pitch_b, yaw_b]).reshape(-1,1)

# Training bounds for xe
pos_err_b = 1
vel_err_b = 0.5
euler_err_b = 0.3  # (rad)

XE_MIN = np.array([-pos_err_b, -pos_err_b, -pos_err_b, -vel_err_b, -vel_err_b, -vel_err_b, -euler_err_b, -euler_err_b, -euler_err_b]).reshape(-1,1)
XE_MAX = np.array([pos_err_b, pos_err_b, pos_err_b, vel_err_b, vel_err_b, vel_err_b, euler_err_b, euler_err_b, euler_err_b]).reshape(-1,1)

# Training bounds for ustar
g = 9.81
accel_b = 3
omega_b = 0.3
UREF_MIN = np.array([-accel_b+g, -omega_b, -omega_b, -omega_b]).reshape(-1,1)
UREF_MAX = np.array([accel_b+g, omega_b, omega_b, omega_b]).reshape(-1,1)

# Training bounds for w (disturbance)
w_accel_b = 2.0
w_att_b = 0.2
w_MIN = np.array([-w_accel_b, -w_accel_b, -w_accel_b, -w_att_b, -w_att_b, -w_att_b]).reshape(-1,1)
w_MAX = np.array([w_accel_b, w_accel_b, w_accel_b, w_att_b, w_att_b, w_att_b]).reshape(-1,1)

### Simulation parameters ###

# Initial error bounds for simulations
pos_0_err_b = 0.25
vel_0_err_b = 0.2
euler_0_err_b = 0.1  # Euler angle initial error (XYZ convention)

XE_INIT_MIN = np.array([-pos_0_err_b, -pos_0_err_b, -pos_0_err_b, -vel_0_err_b, -vel_0_err_b, -vel_0_err_b, -euler_0_err_b, -euler_0_err_b, -euler_0_err_b])
XE_INIT_MAX = np.array([pos_0_err_b, pos_0_err_b, pos_0_err_b, vel_0_err_b, vel_0_err_b, vel_0_err_b, euler_0_err_b, euler_0_err_b, euler_0_err_b])

w_accel_b_sim = 1.0
w_att_b_sim = 0.1
w_sim_MIN = np.array([-w_accel_b_sim, -w_accel_b_sim, -w_accel_b_sim, -w_att_b_sim, -w_att_b_sim, -w_att_b_sim]).reshape(-1,1)
w_sim_MAX = np.array([ w_accel_b_sim,  w_accel_b_sim,  w_accel_b_sim, w_att_b_sim, w_att_b_sim, w_att_b_sim]).reshape(-1,1)

# Time parameters
time_bound = 10*np.pi
time_step = 0.01

def system_reset(seed, trajectory_generator):
    SEED_MAX = 10000000    
    with temp_seed(int(seed * SEED_MAX)):
        xref_0, _ = trajectory_generator(0.)
        xref_0 = xref_0.flatten()
        xe_0 = XE_INIT_MIN + np.random.rand(len(XE_INIT_MIN)) * (XE_INIT_MAX - XE_INIT_MIN)
        
        # For SO3
        Rref_0 = xref_0[6:15].reshape(3, 3)  # Initial reference attitude 
        eul_e_0 = xe_0[6:9]
        Re_0 = euler_xyz_to_rotmat(eul_e_0[0], eul_e_0[1], eul_e_0[2])  # Initial attitude error
        R_0 = Rref_0 @ Re_0  # Initial attitude
        r_0 = R_0.flatten()  

        # For Euclidiean states
        vref_0 = xref_0[0:6]  # Initial reference in vectorspace
        ve_0 = xe_0[0:6]  # Initial error in vectorspace
        v_0 = vref_0 + ve_0  # Initial states in vectorspace

        # Initial state
        x_0 = np.concatenate([v_0, r_0]).reshape(-1,1)

    return x_0
