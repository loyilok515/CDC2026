import torch
import math
import numpy as np

num_dim_x = 18
num_dim_manifold = 12
num_dim_control = 4
num_dim_noise = 3
num_dim_z = 3

J = torch.diag(torch.tensor([0.01, 0.01, 0.03]))

# Skew symmetric matrix from vector
def skew(v):
    """
    Returns the skew-symmetric matrix of a 3D vector.
    """
    bs = v.shape[0]
    return torch.stack([torch.cat([torch.zeros(bs, 1).type(v.type()), -v[:, 2:3], v[:, 1:2]], dim=1),
                        torch.cat([v[:, 2:3], torch.zeros(bs, 1).type(v.type()), -v[:, 0:1]], dim=1),
                        torch.cat([-v[:, 1:2], v[:, 0:1], torch.zeros(bs, 1).type(v.type())], dim=1)], dim=1)


def S_func(x):
    bs = x.shape[0]
    
    # Extract states
    p1, p2, p3, v1, v2, v3, omega1, omega2, omega3, r1, r2, r3, r4, r5, r6, r7, r8, r9 = [x[:, i, 0] for i in range(num_dim_x)]
    
    r = torch.stack([r1, r2, r3, r4, r5, r6, r7, r8, r9], dim=1).reshape(bs, 9, 1)  # (bs, 9, 1)
    R = r.reshape(bs, 3, 3)  # (bs, 3, 3)

    E1 = torch.tensor([[0, 0, 0],
                       [0, 0, -1],    
                       [0, 1, 0]]).type(x.type()).reshape(1, 3, 3).repeat(bs, 1, 1)
    E2 = torch.tensor([[0, 0, 1],
                       [0, 0, 0],    
                       [-1, 0, 0]]).type(x.type()).reshape(1, 3, 3).repeat(bs, 1, 1)
    E3 = torch.tensor([[0, -1, 0],
                       [1, 0, 0],    
                       [0, 0, 0]]).type(x.type()).reshape(1, 3, 3).repeat(bs, 1, 1)
    RE1 = torch.bmm(R, E1).reshape(bs, 9, 1)
    RE2 = torch.bmm(R, E2).reshape(bs, 9, 1)
    RE3 = torch.bmm(R, E3).reshape(bs, 9, 1)

    S = torch.zeros(bs, num_dim_x, num_dim_manifold).type(x.type())
    Sr = 1/np.sqrt(2) * torch.cat([RE1, RE2, RE3], dim=2)
    Somega = torch.eye(3).type(x.type()).repeat(bs, 1, 1)
    Sv = torch.eye(3).type(x.type()).repeat(bs, 1, 1)
    Sp = torch.eye(3).type(x.type()).repeat(bs, 1, 1)
    S[:, 0:3, 0:3] = Sp
    S[:, 3:6, 3:6] = Sv
    S[:, 6:9, 6:9] = Somega
    S[:, 9:18, 9:12] = Sr

    return S


def f_func(x):
    """System dynamics"""
    bs = x.shape[0]
    
    # Extract states
    p1, p2, p3, v1, v2, v3, omega1, omega2, omega3, r1, r2, r3, r4, r5, r6, r7, r8, r9 = [x[:, i, 0] for i in range(num_dim_x)]

    p = torch.stack([p1, p2, p3], dim=1).reshape(bs, 3, 1)  # (bs, 3, 1)
    v = torch.stack([v1, v2, v3], dim=1).reshape(bs, 3, 1)  # (bs, 3, 1)
    omega = torch.stack([omega1, omega2, omega3], dim=1).reshape(bs, 3, 1)  # (bs, 3, 1)
    r = torch.stack([r1, r2, r3, r4, r5, r6, r7, r8, r9], dim=1).reshape(bs, 9, 1)  # (bs, 9, 1)

    R = r.reshape(bs, 3, 3)  # (bs, 3, 3)

    # Gravity
    g = 9.81
    g_I = torch.tensor([0, 0, -g]).type(x.type()).repeat(bs, 1) # (bs, 3)

    # Angular velocity
    J_inv = torch.inverse(J.repeat(bs, 1, 1)).type(x.type())
    cross_term = torch.bmm(skew(omega.reshape(bs, 3)), torch.bmm(J.repeat(bs, 1, 1).type(x.type()), omega))

    # Rotation kinematics
    R_dot = torch.bmm(R, skew(omega.reshape(bs, 3)))
    r_dot = R_dot.reshape(bs, 9)

    f = torch.zeros(bs, num_dim_x, 1).type(x.type())
    f[:, 0:3, 0] = v.reshape(bs, 3)  
    f[:, 3:6, 0] = g_I
    f[:, 6:9, 0] = -torch.bmm(J_inv, cross_term).reshape(bs, 3)
    f[:, 9:18, 0] = r_dot

    return f


def B_func(x):
    """Control input matrix (equivalent to gx from original class)"""
    bs = x.shape[0]
    
    # Extract states
    p1, p2, p3, v1, v2, v3, omega1, omega2, omega3, r1, r2, r3, r4, r5, r6, r7, r8, r9 = [x[:, i, 0] for i in range(num_dim_x)]
    p = torch.stack([p1, p2, p3], dim=1).reshape(bs, 3, 1)  # (bs, 3, 1)
    v = torch.stack([v1, v2, v3], dim=1).reshape(bs, 3, 1)  # (bs, 3, 1)
    omega = torch.stack([omega1, omega2, omega3], dim=1).reshape(bs, 3, 1)  # (bs, 3, 1)
    r = torch.stack([r1, r2, r3, r4, r5, r6, r7, r8, r9], dim=1).reshape(bs, 9, 1)  # (bs, 9, 1)

    R = r.reshape(bs, 3, 3)  # (bs, 3, 3)
    e3 = torch.tensor([0, 0, 1]).type(x.type()).reshape(3, 1).repeat(bs, 1, 1)  # Thrust direction in body frame

    J_inv = torch.inverse(J.repeat(bs, 1, 1)).type(x.type())

    B = torch.zeros(bs, num_dim_x, num_dim_control).type(x.type())
    B[:, 3:6, 0] = torch.bmm(R, e3).reshape(bs, 3)
    B[:, 6:9, 1:4] = J_inv

    return B


def B_w_func(x):
    """noise input to state"""
    bs = x.shape[0]
    
    J_inv = torch.inverse(J.repeat(bs, 1, 1)).type(x.type())

    B_w = torch.zeros(bs, num_dim_x, num_dim_noise).type(x.type())
    B_w[:, 3:6, 0:3] = torch.eye(3).type(x.type()).repeat(bs, 1, 1)
    #B_w[:, 6:9, 3:6] = J_inv
    return B_w

def E_w_func(x):
    bs = x.shape[0]
    
    J_inv = torch.inverse(J.repeat(bs, 1, 1)).type(x.type())

    E_w = torch.zeros(bs, num_dim_manifold, num_dim_noise).type(x.type())
    E_w[:, 3:6, 0:3] = torch.eye(3).type(x.type()).repeat(bs, 1, 1)
    #E_w[:, 6:9, 3:6] = J_inv
    return E_w


def E_func(x):
    bs = x.shape[0]
    # Extract states
    p1, p2, p3, v1, v2, v3, omega1, omega2, omega3, r1, r2, r3, r4, r5, r6, r7, r8, r9 = [x[:, i, 0] for i in range(num_dim_x)]
    
    r = torch.stack([r1, r2, r3, r4, r5, r6, r7, r8, r9], dim=1).reshape(bs, 9, 1)  # (bs, 9, 1)

    R = r.reshape(bs, 3, 3)  # (bs, 3, 3)
    e3 = torch.tensor([0, 0, 1]).type(x.type()).reshape(3, 1).repeat(bs, 1, 1)  # Thrust direction in body frame

    J_inv = torch.inverse(J.repeat(bs, 1, 1)).type(x.type())

    E = torch.zeros(bs, num_dim_manifold, num_dim_control).type(x.type())
    E[:, 3:6, 0] = torch.bmm(R, e3).reshape(bs, 3)
    E[:, 6:9, 1:4] = J_inv
    return E


def Ebot_func(x):
    bs = x.shape[0]
    # Extract states
    p1, p2, p3, v1, v2, v3, omega1, omega2, omega3, r1, r2, r3, r4, r5, r6, r7, r8, r9 = [x[:, i, 0] for i in range(num_dim_x)]

    r = torch.stack([r1, r2, r3, r4, r5, r6, r7, r8, r9], dim=1).reshape(bs, 9, 1)  # (bs, 9, 1)
    R = r.reshape(bs, 3, 3)  # (bs, 3, 3)
    e1 = torch.tensor([1, 0, 0]).type(x.type()).reshape(3, 1).repeat(bs, 1, 1)
    e2 = torch.tensor([0, 1, 0]).type(x.type()).reshape(3, 1).repeat(bs, 1, 1)
    e3 = torch.tensor([0, 0, 1]).type(x.type()).reshape(3, 1).repeat(bs, 1, 1)

    Ebot = torch.zeros(bs, num_dim_manifold, num_dim_manifold - num_dim_control).type(x.type())
    Ebot[:, 0:3, 2:5] = torch.eye(3).type(x.type()).repeat(bs, 1, 1)
    Ebot[:, 3:6, 0] = torch.bmm(R, e1).reshape(bs,3)
    Ebot[:, 3:6, 1] = torch.bmm(R, e2).reshape(bs,3)
    Ebot[:, 9:12, 5:8] = torch.eye(3).type(x.type()).repeat(bs, 1, 1)
    return Ebot

def C_func(x):
    # Compute dz_dx
    bs = x.shape[0]
    C = torch.zeros(bs, num_dim_z, num_dim_x).type(x.type())

    Q_vec = torch.tensor([1, 1, 1])
    Q = torch.diag(Q_vec).repeat(bs, 1, 1).type(x.type())
    C[:, 0:3, 0:3] = Q
    return C

def D_func(x):
    # Compute dz_du
    bs = x.shape[0]
    D = torch.zeros(bs, num_dim_z, num_dim_control).type(x.type())

    R_vec = torch.tensor([0.15, 0.5, 0.5, 0.5])
    R = torch.diag(R_vec).repeat(bs, 1, 1).type(x.type())
    # D[:, 3:7, :] = R
    return D

