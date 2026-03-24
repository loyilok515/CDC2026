import os
import sys
import torch
import importlib


devices = ['cuda', 'cpu'] # save torch model in the format of both devices

task = 'quad_delay'
log = 'log_quad_delay_RCCM'
log_name = log.replace("log_", "", 1)
sys.path.append(log + '/models')
sys.path.append(log + '/systems')
path_to_controller = log + '/controller_best.pth.tar'
system = importlib.import_module('system_'+task)
num_dim_x = system.num_dim_x
num_dim_control = system.num_dim_control

example_x = torch.rand(1,num_dim_x,1)
example_xstar = torch.rand(1,num_dim_x,1)
example_uref = torch.rand(1,num_dim_control,1)
# Saving phase
for device in devices:
    u_func = torch.load(path_to_controller, map_location=torch.device(device), weights_only=False)

    example_x = example_x.to(device)
    example_xstar = example_xstar.to(device)
    example_uref = example_uref.to(device)
    traced = torch.jit.trace(u_func, (example_x, example_xstar, example_uref))
    traced.save(log + '/' + log_name + '_' + device + '.pt')

print("Sucessfully saved neural controller to log")

# Testing phase
test = []

test_x = torch.rand(1,num_dim_x,1)
test_xstar = torch.rand(1,num_dim_x,1)
test_uref = torch.rand(1,num_dim_control,1)
for device in devices:
    controller = torch.jit.load(log + '/' + log_name + '_' + device + '.pt', map_location=device)
    controller.eval()

    x = test_x.to(device)
    xstar = test_xstar.to(device)
    uref = test_uref.to(device)
    test.append(controller(x, xstar, uref).cpu().detach().numpy())

if (test[0] - test[1]).sum() <= 1e-6:
    print("The models are identical on both devices.")
else:
    print("The models differ between devices.")
