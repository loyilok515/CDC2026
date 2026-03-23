import numpy as np
from scipy.io import savemat

log = 'log_quad_noise_RCCM_final'
csv_path = log + '/results/csvs/simulation_data.npz'

# Load the .npz file
data = np.load(csv_path, allow_pickle=True)

# Convert to a regular dictionary (important!)
data_dict = {key: data[key] for key in data.files}

# Save as .mat file
savemat(log+'/results/csvs/simulation_data.mat', data_dict)