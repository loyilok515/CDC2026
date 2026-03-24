import torch

log = 'log_quad'
filename = log+'/model_best.pth.tar'

checkpoint = torch.load(filename, map_location='cpu', weights_only=False)
print("Epoch saved in checkpoint:", checkpoint['epoch'])
print("Best accuracy in checkpoint:", checkpoint['precs'])    
