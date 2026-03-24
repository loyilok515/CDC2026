import numpy as np

def hat(omega):
    """Skew-symmetric matrix from vector"""
    omega=omega.flatten()
    return np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0]
    ])

def vee(M):
    """Inverse of hat map"""
    return np.array([
        M[2,1] - M[1,2],
        M[0,2] - M[2,0],
        M[1,0] - M[0,1]
    ]) * 0.5

def geometric_controller(xcurr, xstar, ustar):
    
    # Parameters
    e3 = np.array([0., 0. , 1.]).reshape(-1,1)
    Kp = np.diag(np.array([0.3, 0.3, 0.6]))
    Kv = np.diag(np.array([1., 1., 3.]))
    Kr = np.diag(np.array([1., 1., 1.]))
    
    # Current states
    pcurr = xcurr[0:3,:]
    vcurr = xcurr[3:6,:]
    Rcurr = xcurr[6:15,:].reshape(3,3)

    # Reference states
    pstar = xstar[0:3,:]
    vstar = xstar[3:6,:]
    Rstar = xstar[6:15,:].reshape(3,3)
    
    # Feedforward input
    astar = ustar[0]
    omegastar = ustar[1:4].reshape(-1,1)

    # Positional control
    err_p = pcurr - pstar
    err_v = vcurr - vstar

    # Thrust
    b3 = Rstar @ e3  # axis 3 in body frame
    a_des = - Kp @ err_p - Kv @ err_v + astar * (b3)
    a = a_des.T @ b3  # Dot product

    # Desired attitude calculations
    z_B_des = a_des / np.linalg.norm(a_des)
    yaw_des = np.arctan2(Rstar[2,1], Rstar[1,1])
    x_C = np.array([np.cos(yaw_des),np.sin(yaw_des),0]).reshape(-1,1)

    y_B_des = hat(z_B_des) @ x_C / (np.linalg.norm(hat(z_B_des) @ x_C) + 1e-8)
    x_B_des = hat(y_B_des) @ z_B_des
    
    R_des = np.hstack([x_B_des, y_B_des, z_B_des])

    # Attitude error
    e_R_mat = 0.5 * (R_des.T @ Rcurr - Rcurr.T @ R_des)
    e_R = vee(e_R_mat).reshape(-1,1)

    # Body rate
    omega = Rcurr.T @ Rstar @ omegastar - Kr @ e_R

    u = np.vstack([a, omega])
    return u


