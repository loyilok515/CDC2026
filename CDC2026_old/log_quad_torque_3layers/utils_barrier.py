import cvxpy as cp
import numpy as np

def barrier_quad(r_vec, u_CCM):
    """
    r_vec: vectorized rotation matrix (9D)
    omega: angular velocity (3D)
    """
    u_barrier = cp.Variable((1, 4))
    tilt_max = 0.35
    alpha = 0.5

    # Barrier function
    r7 = r_vec[6,0]; r8 = r_vec[7,0]; r9 = r_vec[8,0]
    h = r9 - np.cos(tilt_max)
    omega_x = u_barrier[0,1]
    omega_y = u_barrier[0,2]
    r9_dot = -r7*omega_y + r8*omega_x
    h_dot = r9_dot

    constraints = []
    constraints.append(h_dot + alpha*h >= 0)
    
    objective = cp.Minimize(cp.sum_squares(u_barrier - u_CCM.flatten()))

    prob = cp.Problem(objective, constraints)
    prob.solve()

    u_barrier_out = np.array(u_barrier.value).reshape(-1, 1)
    delta_u = u_barrier_out - u_CCM

    return u_barrier_out


def barrier_control_planar_quad(phi, phi_dot):
    """
    phi: roll angle (1D)
    phi_dot: roll rate (1D)
    """
    n = phi.shape[0]
    u2_barrier = cp.Variable((n, 1))
    phi_max = 0.1 #np.pi/6  # 30 degrees
    alpha = 0.5

    constraints = []
    for i in range(n):
        constraints.append(np.sin(phi[i,0])*u2_barrier[i,0] - alpha*(np.cos(phi[i,0]) - np.cos(phi_max)) <= 0)

    objective = cp.Minimize(cp.sum_squares(u2_barrier - phi_dot))

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return u2_barrier.value