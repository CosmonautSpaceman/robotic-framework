fig7 = figure();
subplot(3,1,1),plot(qcol');
grid on;
h1 = title('$$q$$','interpreter','latex');
h1.FontSize=14;
xlabel('x-axis [samples]','interpreter','latex');
ylabel('y-axis [m]','interpreter','latex');

subplot(3,1,2),plot(qdcol');
grid on;
h2 = title('$$\dot{q}$$','interpreter','latex');
h2.FontSize=14;
xlabel('x-axis [samples]','interpreter','latex');
ylabel('y-axis [m]','interpreter','latex');

subplot(3,1,3),plot(aq');
grid on;
h3 = title('$$\ddot{q}$$','interpreter','latex');
h3.FontSize=14;
xlabel('x-axis [samples]','interpreter','latex');
ylabel('y-axis [m]','interpreter','latex');
