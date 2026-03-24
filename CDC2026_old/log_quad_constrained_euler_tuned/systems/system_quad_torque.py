import torch
import math
import numpy as np

num_dim_x = 18
num_dim_manifold = 12
num_dim_control = 4
num_dim_noise = 3

J = torch.tensor([[0.005831, 0, 0],
                  [0, 0.005831, 0],
                  [0, 0, 0.01052]])

# Physical parameters

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
    p1, p2, p3, v1, v2, v3, w1, w2, w3, r1, r2, r3, r4, r5, r6, r7, r8, r9 = [x[:, i, 0] for i in range(num_dim_x)]
    
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
    Sp = torch.eye(3).type(x.type()).repeat(bs, 1, 1)
    Sv = torch.eye(3).type(x.type()).repeat(bs, 1, 1)
    Sw = torch.eye(3).type(x.type()).repeat(bs, 1, 1)
    S[:, 0:3, 0:3] = Sp
    S[:, 3:6, 3:6] = Sv
    S[:, 6:9, 6:9] = Sw
    S[:, 9:18, 9:12] = Sr

    return S


def f_func(x):
    """System dynamics"""
    bs = x.shape[0]
    
    # Extract states
    p1, p2, p3, v1, v2, v3, w1, w2, w3, r1, r2, r3, r4, r5, r6, r7, r8, r9 = [x[:, i, 0] for i in range(num_dim_x)]

    p = torch.stack([p1, p2, p3], dim=1).reshape(bs, 3, 1)  # (bs, 3, 1)
    v = torch.stack([v1, v2, v3], dim=1).reshape(bs, 3, 1)  # (bs, 3, 1)
    w = torch.stack([w1, w2, w3], dim=1).reshape(bs, 3, 1)  # (bs, 3, 1)
    r = torch.stack([r1, r2, r3, r4, r5, r6, r7, r8, r9], dim=1).reshape(bs, 9, 1)  # (bs, 9, 1)

    R = r.reshape(bs, 3, 3)  # (bs, 3, 3)

    # Gravity
    g = 9.81
    g_I = torch.tensor([0, 0, -g]).type(x.type()).repeat(bs, 1) # (bs, 3)

    # Moment of inertia
    J_bs = J.reshape(1, 3, 3).repeat(bs, 1, 1).type(x.type())   # (bs, 3, 3)
    J_inv_bs = J.inverse().reshape(1, 3, 3).repeat(bs, 1, 1).type(x.type())   # (bs, 3, 3)

    # Rotation matrix rows
    r_row1 = torch.stack([r1, r2, r3], dim=1)  # (bs, 3)
    r_row2 = torch.stack([r4, r5, r6], dim=1)  # (bs, 3)
    r_row3 = torch.stack([r7, r8, r9], dim=1)  # (bs, 3)

    f = torch.zeros(bs, num_dim_x, 1).type(x.type())
    f[:, 0:3, 0] = v.reshape(bs, 3)
    f[:, 3:6, 0] = g_I.reshape(bs, 3)
    f[:, 6:9, 0] = -J_inv_bs.bmm(skew(w).bmm(J_bs.bmm(w))).reshape(bs, 3)
    f[:, 9:12, 0] = skew(r_row1).bmm(w).reshape(bs, 3)
    f[:, 12:15, 0] = skew(r_row2).bmm(w).reshape(bs, 3)
    f[:, 15:18, 0] = skew(r_row3).bmm(w).reshape(bs, 3)
    return f


def DfDx_func(x):
    raise NotImplemented('NotImplemented')

def S_f_func(x):
    bs = x.shape[0]
    S_f = torch.zeros(bs, num_dim_manifold, num_dim_manifold).type(x.type())
    return S_f


def B_func(x):
    """Control input matrix (equivalent to gx from original class)"""
    bs = x.shape[0]

    # Extract states
    p1, p2, p3, v1, v2, v3, w1, w2, w3, r1, r2, r3, r4, r5, r6, r7, r8, r9 = [x[:, i, 0] for i in range(num_dim_x)]

    p = torch.stack([p1, p2, p3], dim=1).reshape(bs, 3, 1)  # (bs, 3, 1)
    v = torch.stack([v1, v2, v3], dim=1).reshape(bs, 3, 1)  # (bs, 3, 1)
    w = torch.stack([w1, w2, w3], dim=1).reshape(bs, 3, 1)  # (bs, 3, 1)
    r = torch.stack([r1, r2, r3, r4, r5, r6, r7, r8, r9], dim=1).reshape(bs, 9, 1)  # (bs, 9, 1)

    R = r.reshape(bs, 3, 3)  # (bs, 3, 3)
    e3 = torch.tensor([0, 0, 1]).type(x.type()).reshape(3, 1).repeat(bs, 1, 1)  # Thrust direction in body frame

    # Moment of inertia
    J_inv_bs = J.inverse().reshape(1, 3, 3).repeat(bs, 1, 1).type(x.type())   # (bs, 3, 3)

    B = torch.zeros(bs, num_dim_x, num_dim_control).type(x.type())
    B[:, 3:6, 0] = torch.bmm(R, e3).reshape(bs, 3)
    B[:, 6:9, 1:4] = J_inv_bs

    return B


def DBDx_func(x):
    raise NotImplemented('NotImplemented')

def S_B_func(x):
    bs = x.shape[0]
    
    S_B_1 = torch.zeros(bs, num_dim_manifold, num_dim_manifold).type(x.type())
    S_B_2 = torch.zeros(bs, num_dim_manifold, num_dim_manifold).type(x.type())
    S_B_3 = torch.zeros(bs, num_dim_manifold, num_dim_manifold).type(x.type())

    S_B = torch.stack([S_B_1, S_B_2, S_B_3], dim=3)

    return S_B


def B_w_func(x):
    """noise input to state"""
    bs = x.shape[0]
    
    B_w = torch.zeros(bs, num_dim_x, num_dim_noise).type(x.type())
    B_w[:, 3:6, :] = torch.eye(3).type(x.type()).repeat(bs, 1, 1)
    return B_w

def E_func(x):
    bs = x.shape[0]
    
    # Extract states
    p1, p2, p3, v1, v2, v3, w1, w2, w3, r1, r2, r3, r4, r5, r6, r7, r8, r9 = [x[:, i, 0] for i in range(num_dim_x)]
    
    p = torch.stack([p1, p2, p3], dim=1).reshape(bs, 3, 1)  # (bs, 3, 1)
    v = torch.stack([v1, v2, v3], dim=1).reshape(bs, 3, 1)  # (bs, 3, 1)
    w = torch.stack([w1, w2, w3], dim=1).reshape(bs, 3, 1)  # (bs, 3, 1)
    r = torch.stack([r1, r2, r3, r4, r5, r6, r7, r8, r9], dim=1).reshape(bs, 9, 1)  # (bs, 9, 1)

    R = r.reshape(bs, 3, 3)  # (bs, 3, 3)
    e3 = torch.tensor([0, 0, 1]).type(x.type()).reshape(3, 1).repeat(bs, 1, 1)  # Thrust direction in body frame

    # Moment of inertia
    J_inv_bs = J.inverse().reshape(1, 3, 3).repeat(bs, 1, 1).type(x.type())  # (bs, 3, 3)

    E = torch.zeros(bs, num_dim_manifold, num_dim_control).type(x.type())
    E[:, 3:6, 0] = torch.bmm(R, e3).reshape(bs, 3)
    E[:, 6:9, 1:4] = J_inv_bs
    return E


def Ebot_func(x):
    bs = x.shape[0]
     # Extract states
    p1, p2, p3, v1, v2, v3, w1, w2, w3, r1, r2, r3, r4, r5, r6, r7, r8, r9 = [x[:, i, 0] for i in range(num_dim_x)]

    r = torch.stack([r1, r2, r3, r4, r5, r6, r7, r8, r9], dim=1).reshape(bs, 9, 1)  # (bs, 9, 1)
    R = r.reshape(bs, 3, 3)  # (bs, 3, 3)
    e1 = torch.tensor([1, 0, 0]).type(x.type()).reshape(3, 1).repeat(bs, 1, 1)
    e2 = torch.tensor([0, 1, 0]).type(x.type()).reshape(3, 1).repeat(bs, 1, 1)
    e3 = torch.tensor([0, 0, 1]).type(x.type()).reshape(3, 1).repeat(bs, 1, 1)

    Ebot = torch.zeros(bs, num_dim_manifold, num_dim_manifold - num_dim_control).type(x.type())
    Ebot[:, 0:3, 0:3] = torch.eye(3).type(x.type()).repeat(bs, 1, 1)
    Ebot[:, 3:6, 3] = torch.bmm(R, e1).reshape(bs,3)
    Ebot[:, 3:6, 4] = torch.bmm(R, e2).reshape(bs,3)
    Ebot[:, 9:12, 5:8] = torch.eye(3).type(x.type()).repeat(bs, 1, 1)
    return Ebot