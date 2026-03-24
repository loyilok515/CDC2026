from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import savemat
import numpy as np
from np2pth import get_system_wrapper, get_controller_wrapper
import torch

import importlib
from utils import RK4
from utils_geometric import geometric_controller
import time

import os
import sys

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 13
HUGE_SIZE = 25

# Simulation variables
plot_types = ['2D', '3D', 'time', 'attitude', 'error', 'control', 'control_error', 'disturbance_est_err', 'output_error'] # specify which types of plots to generate; options: '2D', '3D', 'time', 'attitude', 'error', 'control', 'control_error', 'disturbance_err'
plot_dims_dict = {'2D': [0, 1], '3D': [0, 1, 2], 'time': [0, 1, 2], 'attitude': [0, 1, 2], 'error': [0, 1, 2], \
        'control': [1, 2, 3], 'control_error': [1, 2, 3], 'disturbance_est_err': [0, 1, 2], 'output_error': [0, 1, 2, 3, 4, 5, 6]} # specify which state dimensions to plot
nTraj = 1 # Number of trajectories to simulate and plot
sigma = 0.3  # Standard deviation of Gaussian noise added; 
seed = 0  # Random seed for reproducibility

# Configuration variables
task = 'quad'
log = 'log_quad_RCCM' # You'll need to set this to the path of your log model
save_plot_dir = os.path.join(log, 'results/plots')  # Directory to save the plot image, e.g., 'results/plots/3D_path.png'; to show the plots instead, set to None
save_csv_dir = os.path.join(log, 'results/csvs')  # Directory to save the csv files; 
csv_path = os.path.join(save_csv_dir, "simulation_data.npz") # Path to zip folder
mat_path = os.path.join(save_csv_dir, "geo_simulation_data.mat")  # Path to mat folder
simulate = True  # Whether to run the simulation and save data to csv; if False, will load from csv if it exists
UDE_activated = False  # UDE activation flag
time_bound = 15  # Simulation time
time_step = 0.01  # Simulation time step

# Append system, models, config directories from log
if not os.path.isdir(log):
    raise FileNotFoundError(f"{log} does not exist")  # Check if log exists
sys.path.append(log + '/systems')
sys.path.append(log + '/models')
sys.path.append('configs')
sys.path.append('planners')

# Create directory if not exist
if save_plot_dir is not None:
    os.makedirs(save_plot_dir, exist_ok=True)
if save_csv_dir is not None:
    os.makedirs(save_csv_dir, exist_ok=True)

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=HUGE_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=HUGE_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('axes', axisbelow=True)

left = 0.14  # the left side of the subplots of the figure
right = 0.98   # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.925     # the top of the subplots of the figure

np.random.seed(seed)

# Controller
system = importlib.import_module('system_'+task)
f, B, B_w, g, _, num_dim_x, num_dim_manifold, num_dim_control, num_dim_noise = get_system_wrapper(system)
nn_controller = get_controller_wrapper(log + '/controller_best.pth.tar')
controller = nn_controller  # selection of feedback controller (geometric_controller, nn_controller)

# Trajectory generator
planner = importlib.import_module('planner_'+task)
trajectory_generator = planner.forward_spiral_trajectory_generator  # hover, circular, forward_spiral

filename_training = log+'/model_best.pth.tar'
train_data = torch.load(filename_training, map_location='cpu', weights_only=False)
print(f"Loaded model with accuracy {train_data['precs']} from Epoch {train_data['epoch']}") 

if __name__ == '__main__':
    config = importlib.import_module('config_'+task)
    XE_INIT_MIN = config.XE_INIT_MIN
    XE_INIT_MAX = config.XE_INIT_MAX
    w_MIN = config.w_sim_MIN
    w_MAX = config.w_sim_MAX

    t = np.arange(0, time_bound, time_step)
    
    # CSV
    if simulate or (not os.path.exists(csv_path)):
        # States and controls of closed-loop simulation
        x_stars = []
        x_closed = []
        x_att_closed = []
        u_stars = []
        CCM_outputs = []
        controls = []
        errors = []
        d_hat_errors = []
        output_errors = []

        for _ in range(nTraj):
            x_0 = config.system_reset(np.random.rand(), trajectory_generator)
            SO3_index = num_dim_manifold-3  # Index of state vector where SO3 starts
            xstar, x, x_att, ustar, CCM_output, u, d_hat_err, output_error = RK4(controller, trajectory_generator, UDE_activated, SO3_index, f, B, B_w, g, x_0, time_bound, time_step, w_MIN, w_MAX)
            error = [np.sqrt(((xx-xs)**2).sum()) for xx, xs in zip(x, xstar)]

            x_stars.append(xstar)
            x_closed.append(x)
            x_att_closed.append(x_att)
            u_stars.append(ustar)
            CCM_outputs.append(CCM_output)
            controls.append(u)
            errors.append(error)
            d_hat_errors.append(d_hat_err)
            output_errors.append(output_error)

        # Save to zip
        np.savez_compressed(
            csv_path,
            x_stars=x_stars,
            x_closed=x_closed,
            x_att_closed=x_att_closed,
            u_stars=u_stars,
            CCM_outputs=CCM_outputs,
            controls=controls,
            errors=errors, 
            d_hat_errors=d_hat_errors,
            output_errors=output_errors,
        )
        print(f"Saved simulation data to: {csv_path}")

        # Save as .mat file
        data = np.load(csv_path, allow_pickle=True)
        data_dict = {key: data[key] for key in data.files}
        savemat(mat_path, data_dict)
        print(f"Saved simulation data to: {mat_path}")
    
    else:
        # Load data from zip
        data = np.load(csv_path, allow_pickle=True)
        x_stars = data["x_stars"]
        x_closed = data["x_closed"]
        x_att_closed = data["x_att_closed"]
        u_stars = data["u_stars"]
        CCM_outputs = data["CCM_outputs"]
        controls = data["controls"]
        errors=data["errors"]
        d_hat_errors=data["d_hat_errors"]
        output_errors=data["output_errors"]
        print(f"Loaded simulation data from: {csv_path}")

    # Plot
    for plot_type in plot_types:
        plot_dims = plot_dims_dict[plot_type]
        fig = plt.figure(figsize=(8.0, 5.0))
        if plot_type=='3D':
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.gca()

        if plot_type == 'time':
            cmap = plt.get_cmap('plasma')
            colors = [cmap(i) for i in np.linspace(0, 1, len(plot_dims))]
        
        if plot_type == 'attitude':
            cmap = plt.get_cmap('plasma')
            colors = [cmap(i) for i in np.linspace(0, 1, len(plot_dims))]

        if plot_type == 'control':
            cmap = plt.get_cmap('plasma')
            colors = [cmap(i) for i in np.linspace(0, 1, len(plot_dims))]

        if plot_type == 'control_error':
            cmap = plt.get_cmap('plasma')
            colors = [cmap(i) for i in np.linspace(0, 1, len(plot_dims))] 
        
        if plot_type == 'disturbance_est_err':
            cmap = plt.get_cmap('plasma')
            colors = [cmap(i) for i in np.linspace(0, 1, len(plot_dims))]
        
        if plot_type == 'output_error':
            cmap = plt.get_cmap('plasma')
            colors = [cmap(i) for i in np.linspace(0, 1, len(plot_dims))]

        for n_traj in range(nTraj):

            if plot_type=='2D':
                plt.plot([x[plot_dims[0],0] for x in x_closed[n_traj]], [x[plot_dims[1],0] for x in x_closed[n_traj]], 'g', label='closed-loop traj' if n_traj==0 else None)
                plt.plot([x[plot_dims[0],0] for x in x_stars[n_traj]], [x[plot_dims[1],0] for x in x_stars[n_traj]], 'k', label='Reference' if n_traj==0 else None)
                plt.plot(x_stars[n_traj][0][plot_dims[0]], x_stars[n_traj][0][plot_dims[1]], 'ro', markersize=3.)
            elif plot_type=='3D':
                plt.plot([x[plot_dims[0],0] for x in x_closed[n_traj]], [x[plot_dims[1],0] for x in x_closed[n_traj]], [x[plot_dims[2],0] for x in x_closed[n_traj]], 'g', label='closed-loop traj' if n_traj==0 else None)
                plt.plot([x[plot_dims[0],0] for x in x_stars[n_traj]], [x[plot_dims[1],0] for x in x_stars[n_traj]], [x[plot_dims[2],0] for x in x_stars[n_traj]], 'k', label='Reference' if n_traj==0 else None)
                plt.plot(x_stars[n_traj][0][plot_dims[0]], x_stars[n_traj][0][plot_dims[1]], x_stars[n_traj][0][plot_dims[2]], 'ro', markersize=3.)
            elif plot_type=='time':
                for i, plot_dim in enumerate(plot_dims):
                    plt.plot(t, [x[plot_dim,0] for x in x_closed[n_traj]][:-1], color=colors[i])
                    plt.plot(t, [x[plot_dim,0] for x in x_stars[n_traj]][:-1], 'k')
            elif plot_type=='attitude':
                for i, plot_dim in enumerate(plot_dims):
                    plt.plot(t, [x_att[plot_dim,0]*180/np.pi for x_att in x_att_closed[n_traj]], color=colors[i], label=['roll', 'pitch', 'yaw'][plot_dim] if n_traj == 0 else None)
            elif plot_type=='error':
                plt.plot(t, errors[n_traj][:-1], 'g')
            elif plot_type=='control':
                for i, plot_dim in enumerate(plot_dims):
                    plt.plot(t, [u[plot_dim,0] for u in controls[n_traj]], color=colors[i])
                    #plt.plot(t, [u[plot_dim,0] for u in CCM_outputs[n_traj]], 'g--')
                    plt.plot(t, [u[plot_dim,0] for u in u_stars[n_traj]], 'k')
            elif plot_type=='control_error':
                for i, plot_dim in enumerate(plot_dims):
                    plt.plot(t, [u[plot_dim,0] - u_star[plot_dim,0] for u, u_star in zip(controls[n_traj], u_stars[n_traj])], color=colors[i])
            elif plot_type=='disturbance_est_err':
                for i, plot_dim in enumerate(plot_dims):
                    plt.plot(t, [d_hat_err[plot_dim,0] for d_hat_err in d_hat_errors[n_traj]], color=colors[i])
            elif plot_type=='output_error':
                for i, plot_dim in enumerate(plot_dims):
                    plt.plot(t, [output_error[plot_dim,0] for output_error in output_errors[n_traj]], color=colors[i])
                    
        if plot_type=='2D':
            plt.xlabel("x")
            plt.ylabel("y")
            ax.set_aspect('equal', adjustable='box')
        elif plot_type=='3D':
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            # ax.set_aspect('equal', adjustable='box')
        elif plot_type=='time':
            plt.xlabel("t")
            plt.ylabel("x")
        elif plot_type=='attitude':
            plt.xlabel("t")
            plt.ylabel("euler angles (deg)")
        elif plot_type=='error':
            plt.xlabel("t")
            plt.ylabel("error")
        elif plot_type=='control':
            plt.xlabel("t")
            plt.ylabel("u")
        elif plot_type=='control_error':
            plt.xlabel("t")
            plt.ylabel("u - u_star")
        elif plot_type=='disturbance':
            plt.xlabel("t")
            plt.ylabel("d_hat_error")
        elif plot_type=='output_error':
            plt.xlabel("t")
            plt.ylabel("output_error")
        
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
        handles, labels = plt.gca().get_legend_handles_labels()
        if any(labels):
            plt.legend(frameon=True)  # Set legend position here
        if save_plot_dir is not None:
            plt.savefig(save_plot_dir+f'/{task}_{plot_type}.png', dpi=300, bbox_inches='tight')
            print(f"Saved figure to: {save_plot_dir}/{task}_{plot_type}.png")
        else:
            plt.show(block=False)
if save_plot_dir is None:
    plt.show()
