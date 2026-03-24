import numpy as np

# Parameters
g = 9.81

def hover_trajectory_generator(t, dist_est = np.array([[0.], [0.], [0.]])):
    
    # Disturbance estimate from UDE 
    d_hat = dist_est.flatten()
    # xref
    px = 0
    py = 0
    pz = 1
    p = np.array([px, py, pz]).reshape(-1,1)  # Position in inertial frame
    vx = 0
    vy = 0
    vz = 0
    v = np.array([vx, vy, vz]).reshape(-1,1)  # Velocity in inertial frame
    omegax = 0
    omegay = 0
    omegaz = 0
    omega = np.array([omegax, omegay, omegaz]).reshape(-1,1)  # Angular velocity in body frame

    # intermediate variables
    f_vec = d_hat + np.array([0., 0., -g])
    s1 = np.linalg.norm(f_vec)
    s2 = np.sqrt(f_vec[1]**2 + f_vec[2]**2)

    r1 = s2/s1; r2 = 0; r3 = -f_vec[0]/s1
    r4 = -f_vec[1]*f_vec[0]/(s1*s2); r5 = -f_vec[2]/s2; r6 = -f_vec[1]/s1
    r7 = -f_vec[2]*f_vec[0]/(s1*s2); r8 = f_vec[1]/s2; r9 = -f_vec[2]/s1

    R = np.array([[r1, r2, r3], [r4, r5, r6], [r7, r8, r9]])  # Rotation matrix from body frame to inertial frame
    r = np.array([r1, r2, r3, r4, r5, r6, r7, r8, r9]).reshape(-1,1)
    
    xstar_t = np.concatenate([p, v, omega, r]).reshape(-1, 1)

    # uref
    T = s1
    taux = 0
    tauy = 0
    tauz = 0

    ustar_t = np.array([T, taux, tauy, tauz]).reshape(-1,1)

    return xstar_t, ustar_t


def circular_trajectory_generator(t, dist_est = np.array([[0.], [0.], [0.]]), r_c = 1.5, omega = 0.2):

    raise NotImplemented   
    # Disturbance estimate from UDE 
    d_hat = dist_est.flatten()

    # xref
    px = r_c * np.cos(omega * t)
    py = r_c * np.sin(omega * t)
    pz = 1
    p = np.array([px, py, pz]).reshape(-1,1)  # Position in inertial frame
    vx = -omega*r_c * np.sin(omega * t)
    vy = omega*r_c * np.cos(omega * t)
    vz = 0
    v = np.array([vx, vy, vz]).reshape(-1,1)  # Velocity in inertial frame

    # intermediate variables
    s1 = d_hat[0] + omega**2*r_c*np.cos(omega*t)
    s2 = d_hat[1] + omega**2*r_c*np.sin(omega*t)                
    s3 = np.sqrt(s1**2 + s2**2 + (d_hat[2]-g)**2)
    s4 = np.sqrt(s2**2 + (d_hat[2]-g)**2)

    r1 = s4/s3
    r2 = 0
    r3 = -s1/s3
    r4 = -(s1*s2)/(s4*s3)
    r5 = -(d_hat[2]-g)/s4
    r6 = -s2/s3
    r7 = -s1*(d_hat[2]-g)/(s4*s3)
    r8 = s2/s4
    r9 = -(d_hat[2]-g)/s3
    R = np.array([[r1, r2, r3], [r4, r5, r6], [r7, r8, r9]])  # Rotation matrix from body frame to inertial frame
    r = np.array([r1, r2, r3, r4, r5, r6, r7, r8, r9]).reshape(-1,1)

    xstar_t = np.concatenate([p, v, r]).reshape(-1, 1)

    # uref
    T_comp = np.sqrt( (d_hat[0] + omega**2*r_c*np.cos(omega*t))**2 + (d_hat[1] + omega**2*r_c*np.sin(omega*t))**2 + (d_hat[2]-g)**2 )
    p_comp = -(omega**3*r_c*np.cos(omega*t)*(d_hat[2]-g))/(T_comp*np.sqrt( (d_hat[1] + omega**2*r_c*np.sin(omega*t))**2 + (d_hat[2]-g)**2 ))
    q_comp = -p_comp * ( (d_hat[1]**2 + (d_hat[2]-g)**2 + omega**4*r_c**2)*np.sin(omega*t) + (d_hat[1] + d_hat[1]*np.sin(omega*t)**2 + d_hat[0]*np.sin(omega*t)*np.cos(omega*t))*omega**2*r_c + d_hat[0]*d_hat[1]*np.cos(omega*t) ) / (T_comp * np.cos(omega*t) * (d_hat[2]-g))
    r_comp = 0
    ustar_t = np.array([T_comp, p_comp, q_comp, r_comp]).reshape(-1,1)

    return xstar_t, ustar_t


def circular_yaw_aligned_trajectory_generator(t, dist_est = np.array([[0.], [0.], [0.]]), r_c = 1.5, omega = 0.2):
    raise NotImplemented

