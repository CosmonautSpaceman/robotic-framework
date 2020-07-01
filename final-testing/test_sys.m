%%
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

run('new_trajectory.m')


robot2 = SerialLink( [ Revolute('a', 0.7,'m',5,'r',[-0.35; 0; 0], 'I', I1) ...
    Revolute('a', 0.7,'m',5,'r',[-0.35; 0; 0],  'I', I2)],'name', 'my robot')
robot2.gravity = [0; 0; -9.8];

% ts = 0.001
% time_span = 0:ts:1;
le = length(time_span);
ts = 0.00001



qcol = []; qdcol = [];
Jcol = []; Jinfcol = []; 
aq = []; axcol = [];
Jdcol = []; sing_test = []; sing_rank = [];


Fcol = []; 
xc = [0 0 0]; xcd = [0 0 0]; xcdd = [0 0 0];
xdd_out = []; xd_out = []; x_out = [];
xddc_in = []; xdc_in = []; xc_in = [];
F = [0;0;0]
aq_in = [];
qo = []; qdo = []; qddo = [];
countqddr = []; countaq=[]; countax=[];


for i = 1:le
    
    
% Initializiation:
    x = X1(i,:);
    xd = X1dot(i,:);
    xdd = X1ddot(i,:);

    if i == 1
        aq = [aq, [0,0]'];
        aq_in = aq;
%     end
        q = real(calcInverseKin(x));
        qd = calcQd([xd,q']);
        qdcol = [qdcol, qd];
        qcol = [qcol, q];
    else
        qcol = [qcol, q];
        qdcol = [qdcol, qd];
    end
%     
%   Jacobian Testing:
    J = robot2.jacob0(q');
%     J = calcJacobian(q);
    Jcol(:,:,i) = inv(J(1:2,1:2));
    Jinfcol(:,:,i) = J;
    sing_test = [sing_test, det(J(1:2,1:2))];
    sing_rank = [sing_rank, rank(J)];
        
%   acceleration test
    Jinv = inv(J(1:2,1:2));
%     Jd = calcJacobianDot([q qd]);
    Jd = robot2.jacob_dot(q,qd);
%     Jd = Jd(1:length(qd),1:length(qd));
%     Jdcol(:,:,i) = Jd;
    
%     aq = [aq, Jinv*(xdd(1:2) - Jd*qd)];
%     aq1 = calcAq([xdd, q', qd']);  
%     aq = [aq, aq1]
    
%     F = impedanceForceControl([x, xd, xdd, xc, xcd, xcdd]);
%     Fc = F*Kr;  
%     Fcol = [Fcol, F];
%     ax1 = impedanceAccelControl([x,xd,xdd,F',xc, xcd, xcdd]);
%     axcol = [axcol, ax1];
%     
%     ax_temp1 = [axcol(1,1:end-1),ax1(1)];
%     ax_temp2 = [axcol(2,1:end-1),ax1(2)];
%     xd_in1 = trapz(ax_temp1)*ts;
%     xd_in2 = trapz(ax_temp2)*ts;
%     xd_in = [xd_in1;xd_in2];
% 
% %     qd = inv(J(1:2,1:2))*xd_in(1:2,:);
%     if(isnan(qd))
%         qd = [0;0]
%     end
%     
%     xd_temp1 = [axcol(1,1:end-1),xd_in(1)];
%     xd_temp2 = [axcol(2,1:end-1),xd_in(2)];
%     
%     x_in1 = trapz(xd_temp1)*ts;
%     x_in2 = trapz(xd_temp2)*ts;
%     x_in = [xd_in1;xd_in2];
%     
%     xc_in = [xc_in, x_in];
%     xdc_in = [xdc_in, xd_in];
    
%     xddc_in = [xddc_in, xdd_in];
   
    
%     q = real(calcInverseKin(x));
%     if(isnan(q))
%         q = [0;0]
%     end
    
%     aq1 = Jinv*ax1(1:2) + Jd*qd
    aq1 = calcAq([xdd, q', qd']);
    aq_in = [aq_in, aq1];
    
    qd_temp1 = [aq(1,1:end),aq1(1)];
    qd_temp2 = [aq(2,1:end),aq1(2)];
    qd1 = trapz(qd_temp1)*ts;
    qd2 = trapz(qd_temp2)*ts;
    qd = [qd1;qd2];
%     qdo = [qdo, qd];
    
    q_temp1 = [qdcol(1,1:end),qd1];
    q_temp2 = [qdcol(2,1:end),qd2];
    q1 = trapz(q_temp1)*ts;
    q2 = trapz(q_temp2)*ts;
    q = [q1;q2];
%     qo = [qo, qd];

%   Inverse dynamics section    
    tau = double(robot2.rne(q',qd',aq1'));
    
%   Forward dynamics section
    qddr = double(robot2.accel(q', qd' ,tau));

%   Joint Variables (Integration - Post Forward Dynamics)
    qd_temp1 = [aq(1,1:end-1),qddr(1)];
    qd_temp2 = [aq(2,1:end-1),qddr(2)];
    qd1 = trapz(qd_temp1)*ts;
    qd2 = trapz(qd_temp2)*ts;
    qd = [qd1;qd2];
    
    q_temp1 = [qdcol(1,1:end),qd(1)];
    q_temp2 = [qdcol(2,1:end),qd(2)];
    q1 = trapz(q_temp1)*ts;
    q2 = trapz(q_temp2)*ts;
    q = [q1;q2];
    
    aq = [aq, qddr];

%   Conversion to task space again:
    xcdd = computeJoint2Task([q',qd',qddr'])';
    xcd = calcXd([q',qd'])';
    xc = calcForward(q')';   
    
    xdd_out = [xdd_out, xcdd'];
    xd_out = [xd_out, xcd'];
    x_out = [x_out, xc'];
    
% %     qddr = robot2.accel([qc(:,1) qc(:,2)], [qcdot(:,1) qcdot(:,2)] ,[torque(:,1) torque(:,2)])
% 
end

% close all
% figure();
% subplot(3,2,1),plot(xc_in')
% grid on;
% title('x')
% subplot(3,2,3),plot(xdc_in')
% grid on;
% title('xd')
% subplot(3,2,5),plot(axcol')
% grid on;
% title('xdd')
% subplot(3,2,2),plot(X1)
% grid on;
% title('x')
% subplot(3,2,4),plot(X1dot)
% grid on;
% title('xd')
% subplot(3,2,6),plot(X1ddot)
% grid on;
% title('xdd')


% figure();
% plot2(xc_in')
% grid on;
% title('xin')
% 
% figure();
% subplot(2,1,1),plot(Fcol')
% grid on;
% title('F')
% subplot(2,1,2),plot(axcol')
% grid on;
% title('ax in')
% 
% figure()
% subplot(1,2,1),plot(axcol')
% title('ax_{in} (after impedance)')
% grid on
% subplot(1,2,2),plot(aq_in')
% grid on
% title('aq_{in} (before inverse dynamics)')



% qdcol'
% aq';
% Jdcol;
% axcol';

% close all

% figure()
% plot(aq_in')
% title('aq_{in} (before inverse dynamics)')
% grid on
% 
% figure()
% plot(axcol')
% title('ax_{in} (after impedance)')
% grid on



figure()
subplot(1,2,1),plot(aq_in')
title('aq_{in} (before inverse dynamics)')
grid on
subplot(1,2,2),plot(aq')
grid on
title('aq_{out} (after froward dynamics)')

% figure()
% subplot(1,2,1),plot(axcol')
% title('ax_{in} (after impedance)')
% grid on
% subplot(1,2,2),plot(aq_in')
% grid on
% title('aq_{in} (before inverse dynamics)')

figure()
subplot(1,3,1),plot(xdd_out')
title('\ddot{x}')
grid on
subplot(1,3,2),plot(xd_out')
title('\dot{x}')
grid on
subplot(1,3,3),plot(x_out')
title('x')
grid on

figure()
subplot(1,3,1),plot(aq')
title('\ddot{q}')
grid on
subplot(1,3,2),plot(qdcol')
title('\dot{q}')
grid on
subplot(1,3,3),plot(qcol')
title('q')
grid on

figure()
hist(sing_test)
grid on

% 
% figure()
% plot2(x_out')
% hold on
% plot2(X1)
% grid on
