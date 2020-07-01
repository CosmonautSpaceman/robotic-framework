clear all
close all
clc


% Robot Link variables
% m1 = 5; r1 = 0.1; L1 = 0.15;
m2 = 5; r2 = 0.05; L2 = 0.7;
m3 = 5; r3 = 0.05; L3 = 0.7;
% m4 = 4; r4 = 0.03; L4 = 0.4;

% Inertia Matrix
% I_1 = (1/12)*m1*(3*r1^2 + L1^2);
% I_1yy = (1/2)*m1*r1^2;

I_2 = (1/12)*m2*(3*r2^2 + L2^2);
I_2yy = (1/2)*m2*r2^2;

I_3 = (1/12)*m3*(3*r3^2 + L3^2);
I_3yy = (1/2)*m3*r3^2;
% 
% I_4 = (1/12)*m4*(3*r4^2 + L4^2);
% I_4yy = (1/2)*m4*r4^2;


% Remember the order of rigid bodies
I1 = [I_2 0 0; 0 I_2yy 0; 0 0 I_2]
I2 = [I_3 0 0; 0 I_3yy 0; 0 0 I_3]



Q = RNE();
Q = Q;

dh.theta = [0 0 0];
dh.alpha = [0 0 0];
dh.offset = [0 0 0];
dh.d = [0 0 0];
dh.a = [0.7 0.7 0];
dh.type = ['r' 'r' 'r'];

% -------------------------------------------------------------------------
% Rigid body paramaters: inertia, mass, and cener of mass

% Robot Link variables
% m1 = 5; r1 = 0.1; L1 = 0.15;
m2 = 5; r2 = 0.05; L2 = 0.7;
m3 = 5; r3 = 0.05; L3 = 0.7;
% m4 = 4; r4 = 0.03; L4 = 0.4;

% Inertia Matrix
% I_1 = (1/12)*m1*(3*r1^2 + L1^2);
% I_1yy = (1/2)*m1*r1^2;

I_2 = (1/12)*m2*(3*r2^2 + L2^2);
I_2yy = (1/2)*m2*r2^2;

I_3 = (1/12)*m3*(3*r3^2 + L3^2);
I_3yy = (1/2)*m3*r3^2;
% 
% I_4 = (1/12)*m4*(3*r4^2 + L4^2);
% I_4yy = (1/2)*m4*r4^2;


% Remember the order of rigid bodies
I1 = [I_2 0 0; 0 I_2yy 0; 0 0 I_2];
I2 = [I_3 0 0; 0 I_3yy 0; 0 0 I_3];


rb.I =  I1;
rb.I(:,:,2) =  I2;
rb.I(:,:,3) = zeros(3,3);

rb.m = [5 5 0];
rb.r = [-0.35 0 0; -0.35 0 0; 0 0 0]';

% -------------------------------------------------------------------------
% Arbitrary trajectory as the inputs: joint position, velocity, and 
% acceleration
ts = 0.001;
time_span = 0:ts:1;
qc = [pi/3*sin(2*pi*1*time_span)' pi/3*sin(2*pi*1*time_span)'];
qcdot = gradient(qc', ts)';
qcddot = gradient(qcdot', ts)';

figure()
subplot(3,1,1),plot(time_span,qc); grid on; legend('q1','q2','interpreter','latex'); title('q plots','interpreter','latex'); xlabel('time'); ylabel('amplitude');
subplot(3,1,2),plot(time_span,qcdot); grid on; legend('$\dot{q1}$','$\dot{q2}$','interpreter','latex'); title('$\dot{q}$ plots','interpreter','latex'); xlabel('time'); ylabel('amplitude');
subplot(3,1,3),plot(time_span,qcddot); grid on; legend('$\ddot{q1}$','$\ddot{q2}$','interpreter','latex'); title('$\ddot{q}$ plots','interpreter','latex'); xlabel('time'); ylabel('amplitude');

%%

% robot2 = SerialLink( [ Revolute('a', 0.7,'m',5,'r',[-0.35; 0; 0], 'I', I1) ...
%     Revolute('a', 0.7,'m',5,'r',[-0.35; 0; 0],  'I', I2) Revolute('a', 0,'m',0,'r',[0; 0; 0], 'I', zeros(3,3)) ],'name', 'my robot')
robot2 = SerialLink( [ Revolute('a', 0.7,'m',5,'r',[-0.35; 0; 0], 'I', I1) ...
    Revolute('a', 0.7,'m',5,'r',[-0.35; 0; 0],  'I', I2)],'name', 'my robot')
robot2.gravity = [0; 0; -9.8];
torque = robot2.rne(qc,qcdot,qcddot);

% 
figure()

subplot(1,2,1),plot(time_span,Q'); grid on;
legend('\tau_1 computed','\tau_2 computed')
xlabel('time');
ylabel('[Nm]')
title('Computed RNEA')

subplot(1,2,2),plot(time_span,torque); grid on; 
legend('\tau_1 ToolB','\tau_2 ToolB')
xlabel('time');
ylabel('[Nm]')
title('Robotics ToolBox RNEA')

% syms q1 q2 q1d q2d q1dd q2dd a1 a2 d1 d2 m1 m2 r1 r2 alpha1 alpha2

% robot2 = SerialLink( [ Revolute('a', 0.7,'m',5,'r',[0.35; 0; 0]) Revolute('a', 0.7,'m',5,'r',[0.35; 0; 0]) ],'name', 'my robot')
% robot2 = SerialLink( [ Revolute('a',a1,'d',d1,'m',m1,'r',r1) Revolute('a',a2,'d',d2,'m',m2,'r',r2) ],'name', 'my robot')
% robot2 = SerialLink([Revolute('a',a1,'d',d1,'m',m1,'r',r1,'sym') Revolute('a',a1,'d',d1,'m',m1,'r',r1,'sym')], 'name', 'test robot')
% L1 =  Link('d', d1, 'a', a1, 'alpha', alpha1);
% L2 =  Link('d', d2, 'a', a2, 'alpha', alpha2);
% robot = SerialLink([L1 L2],'name','two-link')

% syms a1 a2 g real
% syms c1 c2 m1 m2 real
% syms Ixx1 Iyy1 Izz1 Ixx2 Iyy2 Izz2

% figure()
% plot(torque)
% grid on
% 
% figure()
% plot(Q')


%%








%%

% Finding the inertia matrix M(q)

Mt = robot2.inertia([qc])
M = calc_M();
computed = [];
toolbox = [];

for i = 1:length(time_span)
    computed = [computed M(:,:,i)];
    toolbox = [toolbox Mt(:,:,i)];
end

% Comparing the inertia matrix calculated to the toolbox (Corke)

figure()
subplot(1,2,1),plot(computed(1,:)')
hold on;
subplot(1,2,1),plot(computed(2,:)')
title('Computed M');
xlabel('elements in matrix');
ylabel('[Nm]');
grid on;
hold off

subplot(1,2,2),plot(toolbox(1,:)')
hold on;
subplot(1,2,2),plot(toolbox(2,:)')
title('Toolbox M');
xlabel('elements in matrix');
ylabel('[Nm]');
grid on;
hold off


% Finding the C(q)qdot + G(q) torques
CG = calc_CG();

%%

% Finding the acceleration qdd
qacc = []
for i = 1:length(time_span)
    qacc = [qacc, inv(M(:,:,i))*(Q(1:2,i)- CG(1:2,i))];
end
qddr = robot2.accel([qc(:,1) qc(:,2)], [qcdot(:,1) qcdot(:,2)] ,[torque(:,1) torque(:,2)])


% Printing the comparison:
figure()
subplot(3,1,1),plot(time_span,qacc')
xlabel('time');
ylabel('amplitude');
title('Computed acceleration','interpreter','latex')
legend('$\ddot{q1}$','$\ddot{q2}$','interpreter','latex');
grid on

subplot(3,1,2),plot(time_span,qddr)
xlabel('time');
title('Toolbox acceleration','interpreter','latex')
ylabel('amplitude');
legend('$\ddot{q1}$','$\ddot{q2}$','interpreter','latex');
grid on

subplot(3,1,3),plot(time_span,qcddot); grid on; legend('$\ddot{q1}$','$\ddot{q2}$','interpreter','latex'); title('$\ddot{q}$ original acceleration','interpreter','latex'); xlabel('time'); ylabel('amplitude');

%%
% This part is an old test.

% Iyy1 = 0
% Iyy2 = 0
% b1 = 0
% b2 = 0

% twolink = SerialLink([
%     Revolute('d', 0, 'a', a1, 'alpha', 0, 'm', m1, 'r', [c1 0 0], 'I', [Ixx1 Iyy1 Izz1], 'B', b1, 'G', 1, 'Jm', 0, 'standard')
%     Revolute('d', 0, 'a', a2, 'alpha', 0, 'm', m2, 'r', [c2 0 0], 'I', [Ixx2 Iyy2 Izz2], 'B', b2, 'G', 1, 'Jm', 0, 'standard')
%     ], ...
%     'name', 'two link', ...
%     'comment', 'from Spong, Hutchinson, Vidyasagar');
% twolink = twolink.sym();
% twolink.gravity = [0; 0; g];
% % twolink.base = trotx(sym('pi')/2);
% 
% twolink.inertia([q1 q2])
% twolink.coriolis([q1 q2 q1d q2d])
% twolink.gravload([q1 q2])

