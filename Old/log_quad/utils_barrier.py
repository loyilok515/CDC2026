import cvxpy as cp
import numpy as np

def barrier_control_1D(v, u):
    """
    v: velocity (1D)
    u: control input (1D)
    """
    n = v.shape[0]
    u_barrier = cp.Variable((n, 1))
    v_max = 1.0
    alpha = 2.5

    constraints = []
    for i in range(n):
        constraints.append(2 * v[i,0] * u_barrier[i,0] - alpha * (v_max**2 - v[i,0]**2) <= 0)

    objective = cp.Minimize(cp.sum_squares(u_barrier - u))

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return u_barrier.value


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