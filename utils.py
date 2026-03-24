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


def RK4(controller, trajectory_generator, UDE_activated, SO3_index, f, B, B_w, g, xinit, t_max, dt, w_MIN, w_MAX):
    t = np.arange(0, t_max, dt)

    # Data for states
    trace_star = []
    trace = []
    trace_att = []

    # Data for controls
    u_star = []
    u_CCM = []
    u = []
    
    # Data for UDE disturbance estimation
    d_hat_trace = []
    d_hat_err_trace = []

    # Data for output deviation
    z_err_trace = []

    # initialize states
    xcurr = xinit
    trace.append(xcurr)
    xstar_0, _ = trajectory_generator(0.)
    trace_star.append(xstar_0)
    
    # Initialize quaternion from rotation matrix
    rcurr = xcurr[SO3_index:]
    Rcurr = rcurr.reshape(3, 3)
    rot = Rot.from_matrix(Rcurr)
    qcurr = np.roll(rot.as_quat(), 1) # Change (x, y, z, w) to (w, x, y, z)
    qcurr = np.array(qcurr).reshape(-1,1)

    # Initialize UDE disturbance estimation for force disturbances
    d_hat_init = np.zeros([3,1])
    d_hat_curr = d_hat_init
    integral_term = np.zeros_like(d_hat_init)
    d_hat_trace.append(d_hat_init)


    for i in range(len(t)):
        
        # Generate process noise
        noise = np.zeros([B_w(xinit).shape[1], 1])
        noise += np.array([0.4, 0.4, 0.4, 0.01, 0.01, 0.01]).reshape(-1,1) # Constant noise
        noise += (w_MAX-w_MIN) * np.random.rand(w_MIN.shape[0], 1) + w_MIN # Stochastic noise (uniformly distributed)

        # States in vector space
        vcurr = xcurr[:SO3_index]

        # States in SO3
        rcurr = xcurr[SO3_index:]
        Rcurr = rcurr.reshape(3,3)
        roll, pitch, yaw = rot_to_euler(Rcurr)
        att_curr = np.array([roll, pitch, yaw]).reshape(-1,1)      

        # Get reference trajectory and control
        _t = dt*i  # Current time
        if UDE_activated == True:
            xstar_t, ustar_t = trajectory_generator(_t, dist_est=d_hat_curr)  # Get reference trajectory and control at time t, with disturbance estimate from UDE
        else:
            xstar_t, ustar_t = trajectory_generator(_t)  # Get reference trajectory and control at time t, without disturbance estimate from UDE

        # Runge-Kutta 4 integration
        ui_desired = controller(xcurr, xstar_t, ustar_t)

        # Velocity-based UDE
        ude_gain = 0.5
        velocity_curr = xcurr[3:6]
        g_I = np.array([0., 0., -9.81]).reshape(-1,1) # Gravity vector in inertial frame
        expected_accel = np.array([0., 0., ui_desired[0,0]]).reshape(-1,1) if i>0 else np.zeros_like(d_hat_curr)
        integral_term += - d_hat_curr - (g_I + Rcurr @ expected_accel) if i>0 else np.zeros_like(d_hat_curr) 
        d_hat_next = ude_gain * (integral_term*dt + velocity_curr)
        d_hat_curr = d_hat_next
        d_hat_err = d_hat_curr - noise[0:3]
        # print(f"Time: {_t:.2f}, UDE Disturbance Estimate: {d_hat_curr.flatten()}")
        
        # Actuator dynamics
        # Force dynamics
        tau_a = 0.3 # time constant for acceleration
        accel_desired = ui_desired[0]
        accel_dynamics_activated = False
        if i>0 and accel_dynamics_activated:
            accel_dot = 1/tau_a * (-u[-1][0] + accel_desired)
            accel = u[-1][0] + accel_dot * dt
        else:
            accel = accel_desired 

        # Torque dynamics
        torque_dynamics_activated = False
        if torque_dynamics_activated:
            if i == 0:
                omega = np.zeros([3, 1])
                e_int = np.zeros([3, 1])
            J_pred = np.diag([0.012, 0.011, 0.032]) 
            J = np.diag([0.01, 0.01, 0.03])
            J_inv = np.linalg.inv(J)

            Kp = np.diag([0.1, 0.1, 0.1])
            Kd = np.diag([0.005, 0.005, 0.005])
            Ki = np.diag([0.001, 0.001, 0.001])

            omega_des = ui_desired[1:4]

            # desired angular acceleration (finite difference feedforward)
            omega_des_prev = u_CCM[-1][1:4] if i > 0 else omega_des
            omega_dot_des = (omega_des - omega_des_prev) / dt

            # tracking error
            e = omega_des - omega
            e_dot = (e - e_prev) / dt if i>1 else np.zeros([3, 1])
            e_int += e * dt
            print(f"Time: {_t:.2f}, Omega tracking error: {e.flatten()}")

            tau = J_pred @ omega_dot_des + np.cross(omega_des.squeeze(), (J_pred @ omega_des).squeeze()).reshape(-1, 1) + Kp @ e + Kd @ e_dot # + Ki @ e_int
            omega_dot = J_inv @ (tau - np.cross(omega.squeeze(), (J @ omega).squeeze()).reshape(-1, 1))
            omega = omega + omega_dot * dt
            e_prev = e
        else:
            omega = ui_desired[1:4]

        # Body rate control
        ui = np.vstack([accel, omega])
        omega_b = ui[1:4]
        # Torque control
        # ui = ui_desired
        # omega_b = xcurr[6:9]

        # Output
        zcurr = g(xcurr, ui)
        zstar = g(xstar_t, ustar_t)
        z_err = zcurr-zstar

        # ODE simulation
        vnext = rk4_step(x = xcurr, dt = dt, func = system_dynamics, u = ui, w = noise, f = f, B = B, B_w = B_w)[:SO3_index]
        
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
        u_CCM.append(ui_desired)
        u.append(ui)
        d_hat_trace.append(d_hat_curr)
        d_hat_err_trace.append(d_hat_err)
        z_err_trace.append(z_err)

    return trace_star, trace, trace_att, u_star, u_CCM, u, d_hat_err_trace, z_err_trace


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