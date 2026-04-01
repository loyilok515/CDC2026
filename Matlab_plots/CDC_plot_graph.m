%% Plots for CDC2026
%% Load simulation data
geo_sim_data = load('simulation_data_geo.mat');
CCM_sim_data = load('simulation_data_CCM.mat');
RCCM_sim_data = load('simulation_data_RCCM.mat');
RCCM_UDE_sim_data = load('simulation_data_RCCM_UDE.mat');

%% Position plots
xstar = squeeze(geo_sim_data.x_stars);
geo_x = squeeze(geo_sim_data.x_closed);
CCM_x = squeeze(CCM_sim_data.x_closed);
RCCM_x = squeeze(RCCM_sim_data.x_closed);
RCCM_UDE_x = squeeze(RCCM_UDE_sim_data.x_closed);
f1 = figure(1);
f1.Renderer = 'Painters';
hold on
grid on 
scatter3(geo_x(1,1),geo_x(1,2),geo_x(1,3), DisplayName='Initial position', MarkerFaceColor='r', MarkerEdgeColor='r', LineWidth=1.5)
plot3(xstar(1:30:end,1),xstar(1:30:end,2),xstar(1:30:end,3), 'k--', DisplayName='Reference', LineWidth=2)
plot3(geo_x(:,1),geo_x(:,2),geo_x(:,3), Color='b', DisplayName='Geometric controller')
plot3(CCM_x(:,1),CCM_x(:,2),CCM_x(:,3), Color='r', DisplayName='CCM')
plot3(RCCM_x(:,1),RCCM_x(:,2),RCCM_x(:,3), Color='g', DisplayName='RCCM')
plot3(RCCM_UDE_x(:,1),RCCM_UDE_x(:,2),RCCM_UDE_x(:,3), Color='m', DisplayName='RCCM with UDE')
axis([0 8.25 -4 4.25 -4 4])
daspect([0.5 1 1])
xlabel('x axis ($m$)', 'Interpreter', 'latex', 'FontSize', 13)
ylabel('y axis ($m$)', 'Interpreter', 'latex', 'FontSize', 13)
zlabel('z axis ($m$)', 'Interpreter', 'latex', 'FontSize', 13)
legend('Interpreter','Latex', 'FontSize', 12)
view(3)

%% Output deviation plot
geo_output_err = squeeze(geo_sim_data.output_errors);
geo_output_err_norm = vecnorm(geo_output_err,2,2);
CCM_output_err = squeeze(CCM_sim_data.output_errors);
CCM_output_err_norm = vecnorm(CCM_output_err,2,2);
RCCM_output_err = squeeze(RCCM_sim_data.output_errors);
RCCM_output_err_norm = vecnorm(RCCM_output_err,2,2);
RCCM_UDE_output_err = squeeze(RCCM_UDE_sim_data.output_errors);
RCCM_UDE_output_err_norm = vecnorm(RCCM_UDE_output_err,2,2);

t = linspace(0,15,length(geo_output_err_norm));
alpha = 1.268;
w_bar = 1;
tube_radius = alpha*w_bar;

f2 = figure(2);
f2.Renderer = 'Painters';
ax_main = axes;
axis(ax_main, [0, 15, 0, 3])
hold(ax_main, 'on') 
grid(ax_main, 'on') 

line([0,t(end)], [tube_radius, tube_radius], Color='k', DisplayName='Tube bound $\alpha\overline{w}$', LineWidth=1.5, LineStyle='-.')
plot(t,geo_output_err_norm, Color='b', DisplayName='Geometric controller')
plot(t,CCM_output_err_norm, Color='r', DisplayName='CCM')
plot(t,RCCM_output_err_norm, Color='g', DisplayName='RCCM')
plot(t,RCCM_UDE_output_err_norm, Color='m', DisplayName='RCCM with UDE')

xlabel('Time ($s$)', 'Interpreter','latex', 'FontSize', 13)
ylabel('Output deviation norm $\|\bf{z}(t) - \bf{z}^*(t)\|$', 'Interpreter','latex', 'FontSize', 13)
lgd = legend('Interpreter','Latex', 'FontSize', 12);
lgd.Position = [0.52 0.7 0.2 0.15]; 

x_zoom = [0 0.25];     % t window
y_zoom = [0.5 2.25];   % z range

rectangle('Position', ...
    [x_zoom(1), y_zoom(1), ...
     diff(x_zoom), diff(y_zoom)], ...
    'EdgeColor','k','LineStyle','--')

ax_inset = axes('Position',[0.2 0.7 0.2 0.2]); % [x y w h] in figure

hold(ax_inset,'on')
box(ax_inset,'on')

line([0,t(end)], [tube_radius, tube_radius], Color='k', LineWidth=1.5, LineStyle='-.')
plot(ax_inset, t, geo_output_err_norm,'b')
plot(ax_inset, t, CCM_output_err_norm,'r')
plot(ax_inset, t, RCCM_output_err_norm,'g')
plot(ax_inset, t, RCCM_UDE_output_err_norm,'m')

xlim(ax_inset, x_zoom)
ylim(ax_inset, y_zoom)

rectangle('Position', ...
    [x_zoom(1), y_zoom(1), ...
     diff(x_zoom), diff(y_zoom)], ...
    'EdgeColor','k')

ax_inset.FontSize = 10; 
ax_inset.FontWeight = 'bold';

annotation('arrow',[0.14 0.20],[0.54 0.63])

%% Control input
RCCM_UDE_Tstar = squeeze(RCCM_UDE_sim_data.u_stars(1,:,1));
RCCM_UDE_T = squeeze(RCCM_UDE_sim_data.controls(1,:,1));
RCCM_UDE_omegastar = squeeze(RCCM_UDE_sim_data.u_stars(1,:,2:4));
RCCM_UDE_omega = squeeze(RCCM_UDE_sim_data.controls(1,:,2:4));

f3 = figure(3);
f3.Renderer = 'Painters';
hold on
grid on
axis([0, 15, 6, 14])
plot(t,RCCM_UDE_Tstar, 'k--', DisplayName='Nominal thurst input $f_t^*$', LineWidth=1.5)
plot(t,RCCM_UDE_T, Color='b', DisplayName='RCCM with UDE thrust input $f_t$')

xlabel('Time ($s$)', 'Interpreter','latex', 'FontSize', 13)
ylabel('Mass-normalized thrust input $f_t/m$ $(m/s^{-2})$', 'Interpreter','latex', 'FontSize', 13)
lgd = legend('Interpreter','Latex', 'FontSize', 12);

f4 = figure(4);
f4.Renderer = 'Painters';
hold on
grid on
axis([0, 15, -1, 1])
plot(t,RCCM_UDE_omegastar(:,1), 'r--', DisplayName='Nominal roll rate', LineWidth=1.5)
plot(t,RCCM_UDE_omegastar(:,2), 'g--', DisplayName='Nominal pitch rate', LineWidth=1.5)
plot(t,RCCM_UDE_omegastar(:,3), 'b--', DisplayName='Nominal yaw rate', LineWidth=1.5)
plot(t,RCCM_UDE_omega(:,1), 'r', DisplayName='Roll rate')
plot(t,RCCM_UDE_omega(:,2), 'g', DisplayName='Pitch rate')
plot(t,RCCM_UDE_omega(:,3), 'b', DisplayName='Yaw rate')

xlabel('Time ($s$)', 'Interpreter','latex', 'FontSize', 13)
ylabel('Body angular rate $\omega_B$ $(rad/s)$', 'Interpreter','latex', 'FontSize', 13)
lgd = legend('Interpreter','Latex', 'FontSize', 12);


%% Disturbance Estimation Error
RCCM_UDE_dist_err = squeeze(RCCM_UDE_sim_data.d_hat_errors);

f5 = figure(5);
f5.Renderer = 'Painters';
hold on
grid on
axis([0, 15, -0.5, 1.5])
plot(t,RCCM_UDE_dist_err(:,1), 'r', DisplayName='Force estimate error $f_{d,1}-\hat{f}_{d,1}$')
plot(t,RCCM_UDE_dist_err(:,2), 'g', DisplayName='Force estimate error $f_{d,2}-\hat{f}_{d,2}$')
plot(t,RCCM_UDE_dist_err(:,3), 'b', DisplayName='Force estimate error $f_{d,3}-\hat{f}_{d,3}$')


xlabel('Time ($s$)', 'Interpreter','latex', 'FontSize', 13)
ylabel('Disturbance force estimate error $f_d-\hat{f}_d$ $(N)$', 'Interpreter','latex', 'FontSize', 13)
lgd = legend('Interpreter','Latex', 'FontSize', 12);