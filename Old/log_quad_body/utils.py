import contextlib
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from utils_barrier import barrier_quad

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def RK4(controller, trajectory_generator, f, B, B_w, xinit, t_max, dt, sigma = 0.):
    t = np.arange(0, t_max, dt)

    # Data for states
    trace_star = []
    trace = []
    trace_att = []

    # Data for controls
    u_star = []
    u = []
    
    # d_hat_trace = []
    # d_hat_trace.append(np.zeros([B_w(xinit).shape[1],1]))

    # initialize states
    xcurr = xinit
    trace.append(xcurr)
    xstar_0, _ = trajectory_generator(0.)
    trace_star.append(xstar_0)
    
    # Initialize quaternion from rotation matrix
    rcurr = xcurr[6:15]
    Rcurr = rcurr.reshape(3, 3)
    rot = Rot.from_matrix(Rcurr)
    qcurr = np.roll(rot.as_quat(), 1) # Change (x, y, z, w) to (w, x, y, z)
    qcurr = np.array(qcurr).reshape(-1,1)


    for i in range(len(t)):
        
        # Generate process noise
        noise = np.zeros([B_w(xinit).shape[1], 1])
        # noise = np.array([0.2, -0.3, 0.1]).reshape(-1,1) # Constant noise
        # noise += np.random.randn(B_w(xinit).shape[1], 1) * sigma # Stochastic noise

        # States in vector space
        vcurr = xcurr[0:6]

        # States in SO3
        rcurr = xcurr[6:15]
        Rcurr = rcurr.reshape(3,3)
        roll, pitch, yaw = rot_to_euler(Rcurr)
        att_curr = np.array([roll, pitch, yaw]).reshape(-1,1)      

        # Get reference trajectory and control
        _t = dt*i  # Current time
        xstar_t, ustar_t = trajectory_generator(_t)

        # Runge-Kutta 4 integration
        ui = controller(xcurr, xstar_t, ustar_t)

        ui = barrier_quad(rcurr, ui)

        omega_b = ui[1:4]

        # # First-order lag dynamics
        # omega_b_desired = ui[1:4]
        # omega_b_dot = (u[-1][1:4] - u[-2][1:4])/dt if i>1 else np.zeros((3,1))
        # tau = 0.1
        # omega_b = omega_b_desired - tau * omega_b_dot # omega_b_dot = 1/tau * (- omega_b + omega_b_desired)        

        vnext = rk4_step(x = xcurr, dt = dt, func = system_dynamics, u = ui, w = noise, f = f, B = B, B_w = B_w)[0:6] # [0:9]
        
        qnext = quat_RK4(qcurr, omega_b, K_q=100, dt=dt)
        Rnext = quat_to_dcm(qnext)
        rnext = Rnext.flatten().reshape(-1,1)
        xnext = np.concatenate([vnext, rnext], axis=0)

        # Continue to next time step
        xcurr = xnext
        qcurr = qnext

        # Record data
        trace_star.append(xstar_t)
        trace.append(xcurr)
        trace_att.append(att_curr)  
        u_star.append(ustar_t)
        u.append(ui)

    return trace_star, trace, trace_att, u_star, u


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