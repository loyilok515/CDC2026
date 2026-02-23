import os
import sys
import torch

devices = ['cuda', 'cpu'] # save torch model in the format of both devices

log = 'log_quad_constrained_euler_tuned'
log_name = log.replace("log_", "", 1)
sys.path.append(log + '/models')
path_to_controller = log + '/controller_best.pth.tar'

# Saving phase
for device in devices:
    u_func = torch.load(path_to_controller, map_location=torch.device(device), weights_only=False)

    example_x = torch.tensor([0.2, 0.3, 0.1, 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.]).view(1, -1, 1).to(device)
    example_xstar = torch.tensor([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.]).view(1, -1, 1).to(device)
    example_uref = torch.rand(1,4,1).to(device)
    traced = torch.jit.trace(u_func, (example_x, example_xstar, example_uref))
    traced.save(log + '/' + log_name + '_' + device + '.pt')

print("Sucessfully saved neural controller to log")

# Testing phase
test = []

for device in devices:
    controller = torch.jit.load(log + '/' + log_name + '_' + device + '.pt', map_location=device)
    controller.eval()

    x = torch.tensor([0.2, 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.]).view(1, -1, 1).to(device)
    xstar = torch.tensor([0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.]).view(1, -1, 1).to(device)
    uref = torch.tensor([9.81, 0., 0., 0.]).view(1, -1, 1).to(device)
    test.append(controller(x, xstar, uref).cpu().detach().numpy())

if (test[0] - test[1]).sum() <= 1e-6:
    print("The models are identical on both devices.")
else:
    print("The models differ between devices.")
