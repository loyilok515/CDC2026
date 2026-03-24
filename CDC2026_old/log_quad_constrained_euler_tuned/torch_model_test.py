import torch

devices = ['cuda', 'cpu']
log = 'log_quad'
log_name = log.replace("log_", "", 1)
test = []

for device in devices:
    controller = torch.jit.load(log + '/' + log_name + '_' + device + '.pt', map_location=device)
    controller.eval()

    x = torch.tensor([0.2, 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.]).view(1, -1, 1).to(device)
    xstar = torch.tensor([0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.]).view(1, -1, 1).to(device)
    uref = torch.tensor([9.81, 0., 0., 0.]).view(1, -1, 1).to(device)
    test.append(controller(x, xstar, uref).cpu().detach().numpy())

print(test[0])
if (test[0] - test[1]).sum() <= 1e-6:
    print("The models are identical on both devices.")
else:
    print("The models differ between devices.")