import contextlib
import numpy as np
from scipy.spatial.transform import Rotation as Rot

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def RK4(controller, f, B, B_w, xstar, ustar, xinit, t_max = 10, dt = 0.05, with_tracking = False, sigma = 0.):
    t = np.arange(0, t_max, dt)

    trace = []
    trace_att = []
    u = []
    d_hat_trace = []

    xcurr = xinit
    trace.append(xcurr)
    d_hat_trace.append(np.zeros([B_w(xinit).shape[1],1]))
    # ecurr = np.zeros((3,1))
    # omega_b_lead = np.zeros((3,1))
    # omega_b_curr = np.zeros((3,1))
    # Initialize quaternion from rotation matrix
    rcurr = xcurr[6:15]
    # rcurr = xcurr[9:18]
    Rcurr = rcurr.reshape(3, 3)
    rot = Rot.from_matrix(Rcurr)
    qcurr = np.roll(rot.as_quat(), 1) # Change (x, y, z, w) to (w, x, y, z)
    qcurr = np.array(qcurr).reshape(-1,1)


    for i in range(len(t)):
        
        # Generate process noise
        if with_tracking:
            noise = np.zeros([B_w(xinit).shape[1], 1])
            # noise = np.array([0.2, -0.3, 0.1]).reshape(-1,1) # Constant noise
            # noise += np.random.randn(B_w(xinit).shape[1], 1) * sigma # Stochastic noise
        else:
            noise = np.zeros([B_w(xinit).shape[1], 1])
            # noise = np.array([0.2, -0.3, 0.1]).reshape(-1,1)

        vcurr = xcurr[0:6]
        rcurr = xcurr[6:15]
        # vcurr = xcurr[0:9]
        # rcurr = xcurr[9:18]

        # print(vcurr)
        Rcurr = rcurr.reshape(3,3)
        roll, pitch, yaw = rot_to_euler(Rcurr)
        trace_att.append(np.array([roll, pitch, yaw]).reshape(-1,1))        
        # print(Rcurr)

        # # Simulate noise in the system
        # if with_tracking and i > np.floor(len(t)/2):
        #     ustar[i][1:4] = ustar[i][1:4] + noise
        #     xstar[i][6:9] = xstar[i-1][6:9] + dt*(noise - 1.0 * xstar[i-1][6:9])

        # Runge-Kutta 4 integration
        ui = controller(xcurr, xstar[i], ustar[i]) if with_tracking else ustar[i]
        
        # Positional UDE implementation
        if with_tracking and i>0:
            lambda_d = 1
            x_dot = (trace[-1]-trace[-2])/dt
            d = (x_dot - f(xcurr) - B(xcurr).dot(ui.reshape(-1,1)))[3:6]
            d_hat_prev = d_hat_trace[-1]
            d_hat = d_hat_prev + dt * lambda_d * (-d_hat_prev + d)
            print(d_hat)
            d_hat_trace.append(d_hat)
            # print(d_hat)
            if i<len(t)-1:
                # Compute new (x*, u*) for given disturbance prediction
                # Trajectory generator
                r_c = 1.5
                omega = 0.2 # *0 for straight path
                g = 9.81
                _t = dt*i
                d_hat = d_hat.flatten()
                # u*
                T_comp = np.sqrt( (d_hat[0] + omega**2*r_c*np.cos(omega*_t))**2 + (d_hat[1] + omega**2*r_c*np.sin(omega*_t))**2 + (d_hat[2]-g)**2 )
                p_comp = -(omega**3*r_c*np.cos(omega*_t)*(d_hat[2]-g))/(T_comp*np.sqrt( (d_hat[1] + omega**2*r_c*np.sin(omega*_t))**2 + (d_hat[2]-g)**2 ))
                q_comp = -p_comp * ( (d_hat[1]**2 + (d_hat[2]-g)**2 + omega**4*r_c**2)*np.sin(omega*_t) + (d_hat[1] + d_hat[1]*np.sin(omega*_t)**2 + d_hat[0]*np.sin(omega*_t)*np.cos(omega*_t))*omega**2*r_c + d_hat[0]*d_hat[1]*np.cos(omega*_t) ) / (T_comp * np.cos(omega*_t) * (d_hat[2]-g))
                r_comp = 0
                ustar[i+1] = np.array([T_comp, p_comp, q_comp, r_comp]).reshape(-1,1)
                # x*
                p1 = r_c * np.cos(omega*_t)
                p2 = r_c * np.sin(omega*_t)
                p3 = 1
                v1 = -omega*r_c * np.sin(omega*_t)
                v2 = omega*r_c * np.cos(omega*_t)
                v3 = 0
                xstar[i+1][0:6] = np.array([p1, p2, p3, v1, v2, v3]).reshape(-1,1)
                s1 = d_hat[0] + omega**2*r_c*np.cos(omega*_t)
                s2 = d_hat[1] + omega**2*r_c*np.sin(omega*_t)                
                s3 = np.sqrt(s1**2 + s2**2 + (d_hat[2]-g)**2)
                s4 = np.sqrt(s2**2 + (d_hat[2]-g)**2)
                r1_comp = s4/s3
                r2_comp = 0
                r3_comp = -s1/s3
                r4_comp = -(s1*s2)/(s4*s3)
                r5_comp = -(d_hat[2]-g)/s4
                r6_comp = -s2/s3
                r7_comp = -s1*(d_hat[2]-g)/(s4*s3)
                r8_comp = s2/s4
                r9_comp = -(d_hat[2]-g)/s3
                xstar[i+1][6:15] = np.array([r1_comp, r2_comp, r3_comp, r4_comp, r5_comp, r6_comp, r7_comp, r8_comp, r9_comp]).reshape(-1,1)
    
        omega_b = ui[1:4]

        # First-order lag dynamics
        if with_tracking:
            omega_b_desired = ui[1:4]
            # # Lead compensator here if needed 
            # k = 1.0
            # alpha = 0.5
            # T = 5
            # enext = omega_b_desired - omega_b_curr
            # edot = (enext - ecurr)/dt
            # omega_b_lead_next = omega_b_lead + dt * ((-omega_b_lead/(alpha*T)) + k/alpha*edot + k/(alpha*T)*ecurr)

            # Omega dynamics
            omega_b_dot = (u[-1][1:4] - u[-2][1:4])/dt if i>1 else np.zeros((3,1))
            tau = 0.1
            omega_b = omega_b_desired - tau * omega_b_dot # omega_b_dot = 1/tau * (- omega_b + omega_b_desired)
        # omega_b = ui[4:7]

        # # Soft saturation function
        # if with_tracking:
        #     omega_b = 0.35*np.tanh(omega_b/0.35)           

        vnext = rk4_step(x = xcurr, dt = dt, func = system_dynamics, u = ui, w = noise, f = f, B = B, B_w = B_w)[0:6] # [0:9]
        
        qnext = quat_RK4(qcurr, omega_b, K_q=100, dt=dt)
        Rnext = quat_to_dcm(qnext)
        rnext = Rnext.flatten().reshape(-1,1)
        xnext = np.concatenate([vnext, rnext], axis=0)
        trace.append(xnext)
        u.append(ui)
        xcurr = xnext
        qcurr = qnext
    # u[1] = u[2]
    # u[0] = u[1]
    return trace, trace_att, u


# System dynamics
def system_dynamics(x, u, w, f, B, B_w):
    return f(x) + B(x).dot(u.reshape(-1,1)) + B_w(x).dot(w.reshape(-1,1))


# Get quaternion from dynamics
def quat_RK4(q_curr, omega_b, K_q, dt):
    # Update quarternion
    q_next = rk4_step(x = q_curr, dt = dt, func = Quaternion_Attitude_Dynamics, omega_b = omega_b, K_q = K_q)
    q_next = quat_normalize(q_next)
    return q_next


# Utilities for attitude dynamics
def quat_to_dcm(q):
    q = quat_normalize(q)
    R_IB = R_hat(q) @ np.transpose(L_hat(q))
    return R_IB


# Quaternion attitude dynamics
def Quaternion_Attitude_Dynamics(q, omega_b, K_q):
    q = quat_normalize(q)
    return 0.5*(np.transpose(L_hat(q)) @ omega_b) + ((K_q*(1-np.linalg.norm(q)))*q).reshape(-1, 1)
    

# Define L_hat and R_hat
def L_hat(q):
    q0 = q[0,0]
    q1 = q[1,0]
    q2 = q[2,0]
    q3 = q[3,0]
    L_hat = np.array([[-q1, q0, q3, -q2],
                    [-q2, -q3, q0, q1],
                    [-q3, q2, -q1, q0]])
    return L_hat


def R_hat(q):
    q0 = q[0,0]
    q1 = q[1,0]
    q2 = q[2,0]
    q3 = q[3,0]
    R_hat = np.array([[-q1, q0, -q3, q2],
                    [-q2, q3, q0, -q1],
                    [-q3, -q2, q1, q0]])
    return R_hat


# Quaternion normalization function
def quat_normalize(q):
    q.reshape(-1, 1)
    n = np.linalg.norm(q)
    return np.array([[1.0], [0.0], [0.0], [0.0]]) if n == 0 else q / n

# Rotation matric to Euler angles
def rot_to_euler(Rm):
    # Rm is 3x3
    yaw   = np.arctan2(Rm[1,0], Rm[0,0])
    pitch = np.arctan2(-Rm[2,0], np.sqrt(Rm[2,1]**2 + Rm[2,2]**2))
    roll  = np.arctan2(Rm[2,1], Rm[2,2])
    return roll, pitch, yaw

def euler_xyz_to_rotmat(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr]
    ])

    Ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp]
    ])

    Rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx  # XYZ euler angles

# General Utils

# Generic RK4 Integrator step function
def rk4_step(x, dt, func, *args, **kwargs):
    k1 = func(x,              *args, **kwargs)
    k2 = func(x + 0.5*dt*k1,  *args, **kwargs)
    k3 = func(x + 0.5*dt*k2,  *args, **kwargs)
    k4 = func(x + dt*k3,      *args, **kwargs)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def skew(v):
    """
    Returns the skew-symmetric matrix of a 3D vector.
    """
    v = v.flatten()
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])