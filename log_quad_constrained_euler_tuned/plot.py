from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from np2pth import get_system_wrapper, get_controller_wrapper

import importlib
from utils import RK4, skew
import time

import os
import sys
sys.path.append('systems')
sys.path.append('configs')
sys.path.append('models')

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 13
HUGE_SIZE = 25

# Simulation variables
plot_type = 'error'  # '2D', '3D', 'time', 'attitude', 'error', 'control'
plot_dims = [0, 1, 2]  # For '2D', '3D', 'time' and 'error' plot types, specify which state dimensions to plot
nTraj = 3 # Number of trajectories to simulate and plot
sigma = 0.3  # Standard deviation of Gaussian noise added; 0.3 is set for figure 8 in our paper
seed = 0  # Random seed for reproducibility

# Configuration variables
task = 'quad'
pretrained = 'log_quad_constrained_euler' # 'log_quad_constrained_tuned_sf01_Re_corrected' # You'll need to set this to the path of your pretrained model
save_plot_dir = os.path.join(pretrained, 'results/plots')  # Directory to save the plot image, e.g., 'results/plots/3D_path.png'; to show the plot instead, set to None
save_csv_dir = os.path.join(pretrained, 'results/csvs')  # Directory to save the csv files; disable with None
csv_path = os.path.join(save_csv_dir, "simulation_data.npz") # Path to zip folder
simulate = False

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
plt.rc('legend', fontsize=20)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('axes', axisbelow=True)

left = 0.14  # the left side of the subplots of the figure
right = 0.98   # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.925     # the top of the subplots of the figure

np.random.seed(seed)

system = importlib.import_module('system_'+task)
f, B, B_w, _, num_dim_x, num_dim_manifold, num_dim_control, num_dim_noise = get_system_wrapper(system)
controller = get_controller_wrapper(pretrained + '/controller_best.pth.tar')


if __name__ == '__main__':
    config = importlib.import_module('config_'+task)
    t = config.t
    time_bound = config.time_bound
    time_step = config.time_step
    XE_INIT_MIN = config.XE_INIT_MIN
    XE_INIT_MAX = config.XE_INIT_MAX
    
    # CSV
    if simulate or (not os.path.exists(csv_path)):
        x_0, xstar_0, ustar = config.system_reset(np.random.rand())
        ustar = [u.reshape(-1,1) for u in ustar]
        xstar_0 = xstar_0.reshape(-1,1)
        xstar, _, _ = RK4(None, f, B, B_w, None, ustar, xstar_0, time_bound, time_step, with_tracking=False)  

        # States and controls of closed-loop simulation
        x_closed = []
        x_att_closed = []
        controls = []
        xinits = []
        for _ in range(nTraj):
            xe_0 = XE_INIT_MIN + np.random.rand(len(XE_INIT_MIN)) * (XE_INIT_MAX - XE_INIT_MIN) # Randomly sample XE_init
            # xe_0 = XE_INIT_MIN + np.round(np.random.rand(len(XE_INIT_MIN))) * (XE_INIT_MAX) * 2 # Only plot XE_init_min and XE_init_max

            # For SO3
            xstar_0 = xstar_0.flatten()
            Rref_0 = xstar_0[num_dim_manifold-3:].reshape(3, 3)   
            phi_e_init = xe_0[num_dim_manifold-3:]
            p_e_init = np.linalg.norm(phi_e_init)
            axis_e_init = phi_e_init/p_e_init
            Re_0 = np.cos(p_e_init)*np.eye(3) + (1 - np.cos(p_e_init))*np.outer(axis_e_init, axis_e_init) + np.sin(p_e_init)*skew(axis_e_init)
            R_0 = Rref_0 @ Re_0
            r_0 = R_0.flatten()
            # For Euclidiean states
            vref_0 = xstar_0[0:num_dim_manifold-3]
            ve_0 = xe_0[0:num_dim_manifold-3]
            v_0 = vref_0 + ve_0
            x_0 = np.concatenate([v_0, r_0]).reshape(-1,1)

            xinit = x_0
            xinits.append(xinit)
            x, x_att, u = RK4(controller, f, B, B_w, xstar, ustar, xinit, time_bound, time_step, sigma=sigma, with_tracking=True)
            x_closed.append(x)

            x_att_closed.append(x_att)
            controls.append(u)

        # Save to zip
        np.savez_compressed(
            csv_path,
            xstar=xstar,
            ustar=ustar,
            x_closed=x_closed,
            x_att_closed=x_att_closed,
            controls=controls,
        )
    
    else:
        # Load data from zip
        data = np.load(csv_path, allow_pickle=True)
        xstar = data["xstar"]
        ustar = data["ustar"]
        x_closed = data["x_closed"]
        x_att_closed = data["x_att_closed"]
        controls = data["controls"]

    
    # Plot
    fig = plt.figure(figsize=(8.0, 5.0))
    if plot_type=='3D':
        # ax = fig.gca(projection='3d')
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

    errors = []
    for n_traj in range(nTraj):
        initial_dist = np.sqrt(((x_closed[n_traj][0] - xstar[0])**2).sum())
        errors.append([np.sqrt(((x-xs)**2).sum()) / initial_dist for x, xs in zip(x_closed[n_traj][:-1], xstar)])

        if plot_type=='2D':
            plt.plot([x[plot_dims[0],0] for x in x_closed[n_traj]], [x[plot_dims[1],0] for x in x_closed[n_traj]], 'g', label='closed-loop traj' if n_traj==0 else None)
        elif plot_type=='3D':
            plt.plot([x[plot_dims[0],0] for x in x_closed[n_traj]], [x[plot_dims[1],0] for x in x_closed[n_traj]], [x[plot_dims[2],0] for x in x_closed[n_traj]], 'g', label='closed-loop traj' if n_traj==0 else None)
        elif plot_type=='time':
            for i, plot_dim in enumerate(plot_dims):
                plt.plot(t, [x[plot_dim,0] for x in x_closed[n_traj]][:-1], color=colors[i])
        elif plot_type=='attitude':
            for i, plot_dim in enumerate(plot_dims):
                plt.plot(t, [x_att[plot_dim,0]*180/np.pi for x_att in x_att_closed[n_traj]], color=colors[i], label=['roll', 'pitch', 'yaw'][plot_dim] if n_traj == 0 else None)
        elif plot_type=='error':
            plt.plot(t, [np.sqrt(((x-xs)**2).sum()) for x, xs in zip(x_closed[n_traj][:-1], xstar)], 'g')
        elif plot_type=='control':
            for i, plot_dim in enumerate(plot_dims):
                plt.plot(t, [u[plot_dim,0] for u in controls[n_traj]], color=colors[i])
                
    if plot_type=='2D':
        plt.plot([x[plot_dims[0],0] for x in xstar], [x[plot_dims[1],0] for x in xstar], 'k', label='Reference')
        plt.plot(xstar_0[plot_dims[0]], xstar_0[plot_dims[1]], 'ro', markersize=3.)
        plt.xlabel("x")
        plt.ylabel("y")
        ax.set_aspect('equal', adjustable='box')
    elif plot_type=='3D':
        plt.plot([x[plot_dims[0],0] for x in xstar], [x[plot_dims[1],0] for x in xstar], [x[plot_dims[2],0] for x in xstar], 'k', label='Reference')
        plt.plot(xstar_0[plot_dims[0]], xstar_0[plot_dims[1]], xstar_0[plot_dims[2]], 'ro', markersize=3.)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # ax.set_aspect('equal', adjustable='box')

    elif plot_type=='time':
        for plot_dim in plot_dims:
            plt.plot(t, [x[plot_dim,0] for x in xstar][:-1], 'k')
        plt.xlabel("t")
        plt.ylabel("x")
    elif plot_type=='attitude':
        plt.xlabel("t")
        plt.ylabel("euler angles (deg)")
    elif plot_type=='error':
        plt.xlabel("t")
        plt.ylabel("error")
    elif plot_type=='control':
        for plot_dim in plot_dims:
            plt.plot(t, [u[plot_dim,0] for u in ustar], 'k')
        plt.xlabel("t")
        plt.ylabel("u")
    
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    handles, labels = plt.gca().get_legend_handles_labels()
    if any(labels):
        plt.legend(frameon=True)  # Set legend position here
    if save_plot_dir is not None:
        plt.savefig(save_plot_dir+f'/{task}_{plot_type}.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()
