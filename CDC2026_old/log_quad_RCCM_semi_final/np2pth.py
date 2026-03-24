import os
import sys
import torch
import numpy as np

def get_controller_wrapper(controller_path):
    _controller = torch.load(controller_path, map_location=torch.device('cpu'), weights_only=False) # Change to weights_only=True for PyTorch 2.0+
    _controller.cpu()

    def controller(x, xstar, uref):
        u = _controller(torch.from_numpy(x).float().view(1,-1,1), torch.from_numpy(xstar).float().view(1,-1,1), torch.from_numpy(uref).float().view(1,-1,1)).squeeze(0).detach().numpy()
        return u

    return controller

def get_system_wrapper(system):
    num_dim_x = system.num_dim_x
    num_dim_manifold = system.num_dim_manifold
    num_dim_control = system.num_dim_control
    num_dim_noise = system.num_dim_noise
    num_dim_z = system.num_dim_z
    f_func = system.f_func
    B_func = system.B_func
    B_w_func = system.B_w_func
    g_func = system.g_func
   
    def f(x):
        dot_x = f_func(torch.from_numpy(x).float().view(1,-1,1)).detach().numpy()
        return dot_x.reshape(-1,1)

    def B(x):
        B_value = B_func(torch.from_numpy(x).float().view(1,-1,1)).squeeze(0).detach().numpy()
        return B_value

    def B_w(x):
        B_w_value = B_w_func(torch.from_numpy(x).float().view(1,-1,1)).squeeze(0).detach().numpy()
        return B_w_value

    def g(x, u):
        z = g_func(torch.from_numpy(x).float().view(1,-1,1), torch.from_numpy(u).float().view(1,-1,1)).detach().numpy()
        return z.reshape(-1,1)

    def full_dynamics(x, u):
        return (f(x) + B(x).dot(u.reshape(-1,1))).squeeze(-1)

    return f, B, B_w, g, full_dynamics, num_dim_x, num_dim_manifold, num_dim_control, num_dim_noise
