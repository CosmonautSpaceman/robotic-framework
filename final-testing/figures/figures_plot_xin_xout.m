disp('Plotting Xin and Xout values:')

fig1 = figure();
subplot(3,2,1),plot(x_out');
xlabel('$$x-axis [m]$$','interpreter','latex');
ylabel('$$y-axis [m]$$','interpreter','latex');
grid on;
h1 = title('$$x_{out}$$','interpreter','latex');
h1.FontSize=14;

subplot(3,2,3),plot(xd_out');
xlabel('$$x-axis [m]$$','interpreter','latex');
ylabel('$$y-axis [m]$$','interpreter','latex');
grid on;
h2 = title('$$\dot{x}_{out}$$','interpreter','latex');
h2.FontSize=14;

subplot(3,2,5),plot(xdd_out');
xlabel('$$x-axis [m]$$','interpreter','latex');
ylabel('$$y-axis [m]$$','interpreter','latex');
grid on;
h3 = title('$$\ddot{x}_{out}$$','interpreter','latex');
h3.FontSize=14;

subplot(3,2,2),plot(X1);
xlabel('$$x-axis [m]$$','interpreter','latex');
ylabel('$$y-axis [m]$$','interpreter','latex');
grid on;
h4 = title('$$x_{des}$$','interpreter','latex');
h4.FontSize=14;

subplot(3,2,4),plot(X1dot);
xlabel('$$x-axis [m]$$','interpreter','latex');
ylabel('$$y-axis [m]$$','interpreter','latex');
grid on;
h5 = title('$$\dot{x}_{des}$$','interpreter','latex');
h5.FontSize=14;

subplot(3,2,6),plot(X1ddot);
h6 = title('$$\ddot{x}_{des}$$','interpreter','latex');
h6.FontSize=14;
xlabel('$$x-axis [m]$$','interpreter','latex');
ylabel('$$y-axis [m]$$','interpreter','latex');
grid on;
