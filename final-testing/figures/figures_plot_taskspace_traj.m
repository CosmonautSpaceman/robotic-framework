disp('Plotting task space Trajectory:');
disp('Desired trajectory vs computed traectory');
fig2 = figure()
plot2(x_out');
view(2);
hold on;
plot2(X1);
grid on;
h = title('$$x_{out}$$ Trajectory','interpreter','latex');
h.FontSize=14;
xlabel('x-axis [m]','interpreter','latex');
ylabel('y-axis [m]','interpreter','latex');
legend('Computed Trajectory $$x_{out}$$','Desired Trajectory $$x_{in}$$','interpreter','latex');