import os
import sys
import torch

devices = ['cuda', 'cpu'] # save torch model in the format of both devices

log = 'log_quad'
log_name = log.replace("log_", "", 1)
sys.path.append(log + '/models')
path_to_controller = log + '/controller_best.pth.tar'

for device in devices:
    u_func = torch.load(path_to_controller, map_location=torch.device(device), weights_only=False)

    example_x = torch.tensor([0.2, 0.3, 0.1, 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.]).view(1, -1, 1).to(device)
    example_xstar = torch.tensor([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.]).view(1, -1, 1).to(device)
    example_uref = torch.rand(1,4,1).to(device)
    traced = torch.jit.trace(u_func, (example_x, example_xstar, example_uref))
    traced.save(log + '/' + log_name + '_' + device + '.pt')
