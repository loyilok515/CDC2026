# Neural Robust Control on Lie Groups Using Contraction Methods
Pytorch and Matlab implementation of the CDC2026 paper "Neural Robust Control on Lie Groups Using Contraction Methods", by Yi Lok Lo, Longhao Qian and Hugh H.T. Liu

## Acknowledgement
The codes are developed based on the CoRL'20 paper "[Learning Certified Control Using Contraction Metric](https://arxiv.org/abs/2011.12569)", by Dawei Sun, Susmit Jha, and Chuchu Fan.

## Supplementary documents
The implementation details of the test cases and extra proofs can be found [in this document](Neural_Robust_Control_on_Lie_Groups_Using_Contraction_Methods_(Extended_Version).pdf).

## Requirements
Dependencies include ```torch```, ```tqdm```, ```numpy```, ```scipy```, and ```matplotlib```. You can install them using the following command.
```bash
pip install -r requirements.txt
```

## Dual training of CCM and controller
The script ```main.py``` can be used for learning the controller.

Below is a list of configuration variables and hyperparameters that can be tuned in main.py:

| Hyperparameters | Meaning |
| ------------- | ------------- |
| bs | Batch size |
| num_train | Number of training samples |
| num_test | Number of testing samples |
| learning_rate| Base learning rate |
| epochs | Number of training epochs |
| lr_step | Number of epochs for each learning rate |
| _lambda_fixed | Convergence rate |
| w_ub | Upper bound for dual metric |
| w_lb | Lower bound for dual metric |
| alpha_lb | Lower bound for L-infinity gain |
| epsilon | Small constant to ensure strict inequality |


| Configuration variables | Meaning |
| ------------- | ------------- |
| task | Name of the system |
| log | Directory name for storing the training log |
| metric_type | Train for robust CCM or CCM |
| log_record_file | Text file for training log |
| use_cuda | Enable/disable CUDA |

Run the following command to learn an RCCM controller for the quadrotor system.
```
python3 main.py
```
The neural network model satisfying the RCCM conditions with the lowest training loss will be saved in [[log_name]/model_best.pth.tar](log_quad_RCCM/model_best.pth.tar) and the corresponding learned feedback controller function will be saved in [[log_name]/controller_best.pth.tar](log_quad_RCCM/controller_best.pth.tar). 

## Simulation of closed-loop system
The script ```plot.py``` can be used for simulating the closed-loop system under the learned controller. 

Below is a list of variables to create the desired plot in plot.py:

| Simulation variables | Meaning |
| ------------- | ------------- |
| plot_type | '2D', '3D', 'time', 'attitude', 'error', 'control', 'output_error' |
| plot_dims_dict | Dimensions to be plotted for each plot type |
| nTraj | Number of randomly initialized trajectories to simulate |
| disturbance_switch | Enable/disable constant disturbance force |
| sigma | Bound of Stochastic disturbance force added if disturbance is enabled |

| Configuration variables | Meaning |
| ------------- | ------------- |
| task | Name of the system |
| log | Directory storing pretrained data |
| save_plot_dir | Path to save the plot image |
| save_csv_dir | Path to save the CSV files |
| simulate | Run simulation or plot using saved csv data |
| UDE_switch | Enable/disable UDE | 
| time_bound | Total simulation time |
| time_step | Simulation time step |
| controller | Select trained neural controller/geometric controller |
| trajectory_generator | Select types of trajectories |

Run the following command to evaluate the learned controller and plot the results.
```
python3 plot.py
```

This runs a closed-loop simulation with the learned feedback controller stored in the training log using an RK4 solver. The UDE can be turned on/off to compare the performance of the controller. Plots are saved in [log_quad_RCCM/results/plots](log_quad_RCCM/results/plots) and CSV files are saved in [log_quad_RCCM/results/csvs](log_quad_RCCM/results/csvs) by default. 

## Matlab plots
To visualize the trajectories and create plots used in our paper, run the [MATLAB codes](Matlab_plots/CDC_plot_graph.m) provided in the repository. 

To visualize simulation of your own, the CSV files in .mat format generated in [log_quad_RCCM/results/csvs](log_quad_RCCM/results/csvs) need to be stored in [Matlab_plots](Matlab_plots). 

Change the name of the .mat file at the beginning of the Matlab scripts to the one newly created. 

Finally, run ```CDC_plot_graph.m```.

