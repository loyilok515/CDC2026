import torch
import math
import numpy as np

num_dim_x = 12
num_dim_manifold = 6
num_dim_control = 3
num_dim_noise = 3

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
    r1, r2, r3, r4, r5, r6, r7, r8, r9, v1, v2, v3 = [x[:, i, 0] for i in range(num_dim_x)]
    
    r = torch.stack([r1, r2, r3, r4, r5, r6, r7, r8, r9], dim=1).reshape(bs, 9, 1)  # (bs, 9, 1)
    R = r.reshape(bs, 3, 3)  # (bs, 3, 3)

    # E1 = torch.tensor([[0, 1, 0],
    #                    [-1, 0, 0],
    #                    [0, 0, 0]]).type(x.type()).reshape(1, 3, 3).repeat(bs, 1, 1)
    # E2 = torch.tensor([[0, 0, 1],
    #                    [0, 0, 0],
    #                    [-1, 0, 0]]).type(x.type()).reshape(1, 3, 3).repeat(bs, 1, 1)
    # E3 = torch.tensor([[0, 0, 0],
    #                    [0, 0, 1],
    #                    [0, -1, 0]]).type(x.type()).reshape(1, 3, 3).repeat(bs, 1, 1)

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
    Sv = torch.eye(3).type(x.type()).repeat(bs, 1, 1)
    S[:, 0:9, 3:6] = Sr
    S[:, 9:12, 0:3] = Sv

    return S


def f_func(x):
    """System dynamics"""
    bs = x.shape[0]
    
    # Extract states
    r1, r2, r3, r4, r5, r6, r7, r8, r9, v1, v2, v3 = [x[:, i, 0] for i in range(num_dim_x)]
    f = torch.zeros(bs, num_dim_x, 1).type(x.type())
    r = torch.stack([r1, r2, r3, r4, r5, r6, r7, r8, r9], dim=1).reshape(bs, 9, 1)  # (bs, 9, 1)
    v = torch.stack([v1, v2, v3], dim=1).reshape(bs, 3, 1)  # (bs, 3, 1)
    R = r.reshape(bs, 3, 3)  # (bs, 3, 3)
    
    # System constants
    k = 1.0  # Proportional gain for velocity error
    e = torch.tensor([0, 0, 1]).type(x.type()).reshape(3, 1).repeat(bs, 1, 1)  # Desired direction in body frame

    # Velocity dynamics
    accel = -k*v + torch.bmm(R, e)  # (bs, 3, 1)
    f[:, 9:12, 0] = accel.reshape(bs, 3)
    
    return f


def DfDx_func(x):
    raise NotImplemented('NotImplemented')

def S_f_func(x):
    bs = x.shape[0]
    
    # Extract states
    r1, r2, r3, r4, r5, r6, r7, r8, r9, v1, v2, v3 = [x[:, i, 0] for i in range(num_dim_x)]
    
    r = torch.stack([r1, r2, r3, r4, r5, r6, r7, r8, r9], dim=1).reshape(bs, 9, 1)  # (bs, 9, 1)
    R = r.reshape(bs, 3, 3)  # (bs, 3, 3)

    # System constants
    k = 1.0  # Proportional gain for velocity error
    e = torch.tensor([0, 0, 1]).type(x.type()).reshape(3, 1)  # Desired direction in body frame
    
    S_f = torch.zeros(bs, num_dim_manifold, num_dim_manifold).type(x.type())

    # E1 = torch.tensor([[0, 1, 0],
    #                    [-1, 0, 0],
    #                    [0, 0, 0]]).type(x.type()).reshape(1, 3, 3).repeat(bs, 1, 1)
    # E2 = torch.tensor([[0, 0, 1],
    #                    [0, 0, 0],
    #                    [-1, 0, 0]]).type(x.type()).reshape(1, 3, 3).repeat(bs, 1, 1)
    # E3 = torch.tensor([[0, 0, 0],
    #                    [0, 0, 1],
    #                    [0, -1, 0]]).type(x.type()).reshape(1, 3, 3).repeat(bs, 1, 1)

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

    Sr = 1/np.sqrt(2) * torch.cat([RE1, RE2, RE3], dim=2)

    Sv = torch.eye(3).type(x.type()).repeat(bs, 1, 1)
    S_f[:, 0:3, 0:3] = -k * Sv
    S_f[:, 0:3, 3:6] = torch.bmm(torch.kron(torch.eye(3).type(x.type()), e.transpose(0, 1)).repeat(bs, 1, 1), Sr)
    return S_f


def B_func(x):
    """Control input matrix (equivalent to gx from original class)"""
    bs = x.shape[0]
    # Extract states
    r1, r2, r3, r4, r5, r6, r7, r8, r9, v1, v2, v3 = [x[:, i, 0] for i in range(num_dim_x)]
    B = torch.zeros(bs, num_dim_x, num_dim_control).type(x.type())
    r = torch.stack([r1, r2, r3, r4, r5, r6, r7, r8, r9], dim=1).reshape(bs, 9, 1)  # (bs, 9, 1)
    R = r.reshape(bs, 3, 3)  # (bs, 3, 3)
    
    r_row1 = torch.stack([r1, r2, r3], dim=1)  # (bs, 3)
    r_row2 = torch.stack([r4, r5, r6], dim=1)  # (bs, 3)
    r_row3 = torch.stack([r7, r8, r9], dim=1)  # (bs, 3)
    
    B[:, :, :] = torch.cat([skew(r_row1), skew(r_row2), skew(r_row3), torch.zeros(bs, 3, 3).type(x.type())], dim=1)  # (bs, 12, 3)

    return B


def DBDx_func(x):
    raise NotImplemented('NotImplemented')

def S_B_func(x):
    bs = x.shape[0]
    
    # Extract states
    r1, r2, r3, r4, r5, r6, r7, r8, r9, v1, v2, v3 = [x[:, i, 0] for i in range(num_dim_x)]
    

    S_B_1 = torch.zeros(bs, num_dim_manifold, num_dim_manifold).type(x.type())
    F_1 = 0.5*torch.stack([torch.stack([r2-r4,torch.zeros(bs).type(x.type()),torch.zeros(bs).type(x.type())],dim=1), 
                         torch.stack([r3, -r4, r1],dim=1),  
                         torch.stack([r6, -r5, r2],dim=1)], dim=2)
    S_B_1[:, 3:6, 3:6] = F_1

    S_B_2 = torch.zeros(bs, num_dim_manifold, num_dim_manifold).type(x.type())
    F_2 = 0.5*torch.stack([torch.stack([-r7, r2, -r1],dim=1), 
                         torch.stack([torch.zeros(bs).type(x.type()), r3-r7, torch.zeros(bs).type(x.type())],dim=1),  
                         torch.stack([r9, -r8, r3],dim=1)], dim=2)
    S_B_2[:, 3:6, 3:6] = F_2

    S_B_3 = torch.zeros(bs, num_dim_manifold, num_dim_manifold).type(x.type())
    F_3 = 0.5*torch.stack([torch.stack([-r8, r5, -r4],dim=1), 
                         torch.stack([-r9, r6, -r7],dim=1),  
                         torch.stack([torch.zeros(bs).type(x.type()), torch.zeros(bs).type(x.type()), r6-r8],dim=1)], dim=2)
    S_B_3[:, 3:6, 3:6] = F_3

    S_B = torch.stack([S_B_1, S_B_2, S_B_3], dim=3)

    return S_B


def B_w_func(x):
    """noise input to state"""
    bs = x.shape[0]
    
    B_w = torch.zeros(bs, num_dim_x, num_dim_noise).type(x.type())
    return B_w

def E_func(x):
    bs = x.shape[0]
    E = torch.zeros(bs, num_dim_manifold, num_dim_control).type(x.type())
    E[:, 0:3, :] = np.sqrt(2) * torch.eye(3).type(x.type()).repeat(bs, 1, 1)
    return E


def Ebot_func(x):
    bs = x.shape[0]
    Ebot = torch.zeros(bs, num_dim_manifold, num_dim_manifold - num_dim_control).type(x.type())
    Ebot[:, 3:6, :] = np.sqrt(2) * torch.eye(3).type(x.type()).repeat(bs, 1, 1)
    return Ebot