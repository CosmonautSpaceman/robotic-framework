% clear all
% close all
% clc



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


robot2 = SerialLink( [ Revolute('a', 0.7,'m',5,'r',[-0.35; 0; 0], 'I', I1) ...
    Revolute('a', 0.7,'m',5,'r',[-0.35; 0; 0],  'I', I2)],'name', 'my robot')
robot2.gravity = [0; 0; -9.8];
robot2.gravity = [0; 0; 0];

q = [1 1]
qd = [2 2]
qdd = [3 3]

torque = robot2.rne(q,qd,qdd)


%%


% sol = robot2.ikine_sym(2)
sol = robot2.ikine_sym(2)
length(sol)
s1 = sol{1};  % is one solution
q1 = s1(1)      % the expression for q1
q2 = s1(2)      % the expression for q2



syms q1 q2 real
syms ty tx real


TE = robot2.fkine([q1,q2])
e1 = tx == TE.t(1)
e2 = ty == TE.t(2)
[s1,s2] = solve( [e1 e2], [q1 q2] );
simplify(s1)
simplify(s2)


robot2.coriolis([0.35,0.35],[1,1])
%%


%%
% robot.workspace
% Seg 1
robot = SerialLink( [ Revolute('d',0.15,'a', 0.7) Revolute('a', 0.7) ],'name', 'my robot')
robot.plot([0,0],'jvec','arrow','zoom',0.7)
title('Kinematic Model')
xlabel('X [m]')
ylabel('Y [m]')
zlabel('Z [m]')

p1 = linspace(0,pi,100)
p2 = linspace(0,deg2rad(150),100)
p3 = linspace(pi,0,100)
p4 = linspace(deg2rad(150),deg2rad(-150),100)

Tcollect = []
seg1 = []
seg2 = []
seg3 = []
seg4 = []
seg5 = []

for i= 1:length(p1)
    [T,A] = robot2.fkine([p1(i), 0]);
    segrange = A(2).transl;
    seg1 = [segrange' seg1];
end
% Seg 2
for i= 1:length(p1)
    [T,A] = robot2.fkine([p1(end),p2(i)]);
    segrange = A(2).transl;
    seg2 = [segrange' seg2];
end
% Seg 3
for i= 1:length(p1)
    [T,A] = robot2.fkine([p3(i),p2(end)]);
    segrange = A(2).transl;
    seg3 = [segrange' seg3];
end
% Seg 4
for i= 1:length(p1)
    [T,A] = robot2.fkine([p3(end),p4(i)]);
    segrange = A(2).transl;
    seg4 = [segrange' seg4];
end
% Seg 5
for i= 1:length(p1)
    [T,A] = robot2.fkine([p1(i),p4(end)]);
    segrange = A(2).transl;
    seg5 = [segrange' seg5];
end

seg1 = seg1'
seg2 = seg2'
seg3 = seg3'
seg4 = seg4'
seg5 = seg5'

seg_total = [seg1;seg2;seg3;seg4;seg5]

% Plot without the robot
figure();
% robot2.plot([0,0],'jvec','arrow','workspace', [-1.5, 1.5 -1 1.5 -1 1],'noname')
% hold on
plot2(seg1,'color',[0 0.4470 0.7410])
hold on
plot2(seg2,'color',[0 0.4470 0.7410])
hold on
plot2(seg3,'color',[0 0.4470 0.7410])
hold on
plot2(seg4,'color',[0 0.4470 0.7410])
hold on
plot2(seg5,'color',[0 0.4470 0.7410])
hold on
grid on
title('Reachable Workspace (Max/min)')
hold on
plot2(X1,'ro')
% a = legend('Desired Trajectory')
hold on
plot2(x_out','b.')
% b = legend('Computed Trajectory')
% legend(a,b)
xlabel('x-axis [m]')
ylabel('y-axis [m]')


% Plot with the robot
figure();
robot2.plot([0,0],'jvec','arrow','workspace', [-1.5, 1.5 -1 1.5 -1 1],'noname')
hold on
plot2(seg1,'color',[0 0.4470 0.7410])
hold on
plot2(seg2,'color',[0 0.4470 0.7410])
hold on
plot2(seg3,'color',[0 0.4470 0.7410])
hold on
plot2(seg4,'color',[0 0.4470 0.7410])
hold on
plot2(seg5,'color',[0 0.4470 0.7410])
hold on
grid on
title('Reachable Workspace (Max/min)')
hold on
plot2(X1,'ro')
% a = legend('Desired Trajectory')
hold on
plot2(x_out','b.')
% b = legend('Computed Trajectory')
% legend(a,b)
xlabel('x-axis [m]')
ylabel('y-axis [m]')

%%
% ts = 0.001
% time_span = 0:ts:3;
% le = length(time_span);

% p1 = linspace(0,pi,length(time_span))
% seg1 = []
% for i= 1:length(p1)
%     [T,A] = robot2.fkine([p1(i), p1(i)]);
%     segrange = A(2).transl;
%     seg1 = [segrange' double(seg1)];
% end
% X1 = double(seg1')
% x = linspace(-1,1,le)
% % y = linspace(0.8,1.3,le)
% y = linspace(0.5,1,le)
% z = zeros(1,le)

% X1 = [x',y',z']
% X1dot = gradient(X1', ts)'
% X1ddot = gradient(X1dot', ts)'


figure();
% robot2.plot([0,0],'jvec','arrow','workspace', [-1.5, 1.5 -1 1.5 -1 1],'noname')
% hold on
plot2(seg1,'color',[0 0.4470 0.7410])
hold on
plot2(seg2,'color',[0 0.4470 0.7410])
hold on
plot2(seg3,'color',[0 0.4470 0.7410])
hold on
plot2(seg4,'color',[0 0.4470 0.7410])
hold on
plot2(seg5,'color',[0 0.4470 0.7410])
hold on
grid on
title('Reachable Workspace (Max/min)')
hold on
plot2(X1,'ro')
% a = legend('Desired Trajectory')
hold on
plot2(x_out','b.')
% b = legend('Computed Trajectory')
% legend(a,b)
xlabel('x-axis [m]')
ylabel('y-axis [m]')




%%
[x,xd,xdd] = tpoly(-1,1,le);
[y,yd,ydd] = tpoly(0.5,1,le);
z = zeros(le,1);
zd = zeros(le,1);
zdd = zeros(le,1);


X1 = [x,y,z];
X1dot = [xd,yd,zd];
X1ddot = [xdd,ydd,zdd];


%%
qcol = [];
for i = 1:length(X1)
    x = X1(i,:);  
    q = real(calcInverseKin(x));
    qcol = [qcol, q]
end
qcol = qcol'
q_s = qcol(1,:)
q_f = qcol(end,:)

Ts = robot2.fkine(q_s)
Tf = robot2.fkine(q_f)

T = ctraj(Ts,Tf,le)
figure()
plot(time_span, T.transl)

figure()
plot2(T.transl)
grid on
%%

t0 = 0; tf = 30; syms q0; v0 = 0; a0 = 0; syms  qf;  vf = 0;  af = 0; samples = 3000
% KP = (eye(2).*3); 
% KD = eye(2).*5;

run('new_trajectory.m');
% run('new_trajectory2.m');
close all
% % Simulink Trajectory
% Xt = [X1;X2]
% Xtdot = [X1dot;X2dot]
% Xtddot = [X1ddot;X2ddot]

Xt = [X1]
Xtdot = [X1dot]
Xtddot = [X1ddot]

time_span = linspace(0,tf,samples);
X1 = [time_span', Xt]
X1dot = [time_span', Xtdot]
X1ddot = [time_span', Xtddot]

simOut = sim('impedanceWithTorque')
pF_ext = simOut.F_ext.signals.values
ptau = simOut.tau.signals.values
pF_ext_time = simOut.time

pq_out = simOut.q_out.signals.values;
pqd_out = simOut.qd_out.signals.values;
pqdd_out = simOut.qdd_out.signals.values;

x_out = simOut.x_out.signals.values;
xd_out = simOut.xd_out.signals.values;
xdd_out = simOut.xdd_out.signals.values;

F_ext = [];
tau = [];

q_out = [];
qd_out = [];
qdd_out = [];

% x_out = [];
% xd_out = [];
% xdd_out = [];

for i = 1:6001
%     F_ext = [F_ext, pF_ext(:,:,i)];
%     tau = [tau, ptau(:,:,i)];
    
    q_out = [q_out, pq_out(:,:,i)];
    qd_out = [qd_out, pqd_out(:,:,i)];
    qdd_out = [qdd_out, pqdd_out(:,:,i)];
end

% F_ext = F_ext';
% tau = tau';

q_out = q_out';
qd_out = qd_out';
qdd_out = qdd_out';

figure()
subplot(3,1,1),plot(q_out);
title('q_{out} (Simulink simOut)')
grid on;
subplot(3,1,2),plot(qd_out);
title('qd_{out} (Simulink simOut)')
grid on;
subplot(3,1,3),plot(qdd_out);
title('qdd_{out} (Simulink simOut)')
grid on;

figure()
subplot(3,1,1),plot(x_out);
title('x_{out} (Simulink simOut)')
grid on;
subplot(3,1,2),plot(xd_out);
title('xd_{out} (Simulink simOut)')
grid on;
subplot(3,1,3),plot(xdd_out);
title('xdd_{out} (Simulink simOut)')
grid on;

% figure()
% subplot(2,1,1),plot(F_ext);
% title('F_{ext} (Simulink simOut)')
% grid on;
% subplot(2,1,2),plot(tau);
% title('\tau')
% grid on;

figure();plot2(x_out); 
hold on; plot2(Xt); 
grid on; 
title('Desired vs computed trajectory'); 
xlabel('x-axis [m]'); 
ylabel('y-axis [m]');

%%
clear all
close all
clc

run('robot_init.m');

tf0 = 10:10:100;
vf0 = zeros(1,length(tf0));
af0 =  zeros(1,length(tf0));
samp = ones(1,length(tf0))*3000;
ForceTest = ones(1,length(tf0))*false;
ForceImpulse = ones(1,length(tf0))*true;

% Generating control matries

kdp = 1:2:9
bdp = 1:2:9

for i = 1:10
    if(i<6)
        KdI(:,:,i) = eye(3).*kdp(i);
        BdI(:,:,i) = eye(3).*5;
    else
        KdI(:,:,i) = eye(3).*3;
        BdI(:,:,i) = eye(3).*bdp(i-5);
    end
end

KdIn = [eye(2).*((0.01)^2) zeros(2,1); zeros(1,3)]
KpIn = [eye(2).*(0.01)*2 zeros(2,1); zeros(1,3)]

% for k = 1:length(tf0)
for k = 1:1
    t0 = 0; tf = 30; syms q0; v0 = 0; a0 = 0; syms  qf;  vf = vf0(k);  af = af0(k); samples = samp(k)
    
%     Kd = KdI(:,:,k)
%     Bd = BdI(:,:,k)
    Kd =  eye(3).*5;
    Bd = eye(3).*7;
    Md = inv(eye(2).*(1));

    run('new_trajectory.m');
    run('new_trajectory2.m');
    % % Simulink Trajectory
    X1 = [X1;X2];
    X1dot = [X1dot;X2dot];
    X1ddot = [X1ddot;X2ddot];
    le = length(time_span)*2;

    % Variable vectors/matrices.Kd
    qcol = []; qdcol = [];
    Jcol = []; Jinfcol = []; 
    aq = []; axcol = [];
    Jdcol = []; sing_test = []; sing_rank = [];

    Fcol = []; 
    xc = [0 0 0]; xcd = [0 0 0]; xcdd = [0 0 0];
    xdd_out = []; xd_out = []; x_out = [];
    xddc_in = []; xdc_in = []; xc_in = [];
    F = [0;0;0];

    aq_in = [];
    qo = []; qdo = []; qddo = [];
    countqddr = []; countaq=[]; countax=[];


    % The main control loop:
    for i = 1:le
    %   Initializiation:
        x = X1(i,:);
        xd = X1dot(i,:);
        xdd = X1ddot(i,:);

    %   Initialization of the joint variables.
        if i == 1
            aq = [aq, [0,0]'];
            aq_in = aq;
            q = real(calcInverseKin(x));
            qd = calcQd([xd,q']);
            qdcol = [qdcol, qd];
            qcol = [qcol, q];
        else
            qcol = [qcol, q];
            qdcol = [qdcol, qd];
        end

    %   Force disturbance input:
    
        if (ForceTest(k) == true)
            if(i > 250)
                F = [1;1;0];
            end
        end
        
        if (ForceImpulse(k) == true)
            if(i > 250) && (i < 700)
                F = [1;1;0];
            else
                F = [0;0;0];
            end
        end

    %   Impedance control section:
        Fcol = [Fcol, F];
        ax1 = impedanceAccelControl2([x,xd,xdd,F',xc, xcd], Md,Bd,Kd);
        axcol = [axcol, ax1];
        
        ax_temp1 = [axcol(1,1:end-1),ax1(1)];
        ax_temp2 = [axcol(2,1:end-1),ax1(2)];
        xd_in1 = trapz(ax_temp1)*ts;
        xd_in2 = trapz(ax_temp2)*ts;
        xd_in = [xd_in1;xd_in2];

        xd_temp1 = [axcol(1,1:end-1),xd_in(1)];
        xd_temp2 = [axcol(2,1:end-1),xd_in(2)];

    %   The integrated trajectory into the forward dynamics system in task
    %   space
        x_in1 = trapz(xd_temp1)*ts;
        x_in2 = trapz(xd_temp2)*ts;
        x_in = [xd_in1;xd_in2];

        xc_in = [xc_in, x_in];
        xdc_in = [xdc_in, xd_in];
        
        close all
        clc
        
        e = [x_in;0] - xc'
        ed = [xd_in;0] - xcd'
        w = ax1 + KdIn*ed + KpIn*e
        
        Jdq = calcJacobianDot([q(1), q(2), qd(1), qd(2)]) * qd
        J = calcJacobian(q);
        J = J(1:2,1:2);
        
        lambda = 0.1;
        Jinv = (J' * inv(J*J'+eye(2)*lambda))';
%         TestSingularity = det(J(1:2,1:2));
            
        AQ = Jinv * (w(1:2) - Jdq(1:2))
        
    %   Conversion from task-space to joint space  
%         aq1 = calcQdd([ax1', q', qd']);
        aq1 = AQ(1:2)
        aq_in = [aq_in, aq1];

    %   Joint variable integration
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
%         tau = double(robot2.rne(q',qd',aq1'));
        tau = RNE_sim2([q', qd', aq1']);



%         tau = RNE_sim2([q', qd', aq1']);

    %   Forward dynamics section
%         qddr = double(robot2.accel(q', qd' ,tau));
        qddr = forwardDyn([tau', q', qd']);

    %   Joint Variables (Integration - Post Forward Dynamics)
    %   Integration: Joint velocity from Joint acceleration.
        qd_temp1 = [aq(1,1:end-1),qddr(1)];
        qd_temp2 = [aq(2,1:end-1),qddr(2)];
        qd1 = trapz(qd_temp1)*ts;
        qd2 = trapz(qd_temp2)*ts;
        qd = [qd1;qd2];

    %   Integration: Joint position from Joint velocity:    
        q_temp1 = [qdcol(1,1:end),qd(1)];
        q_temp2 = [qdcol(2,1:end),qd(2)];
        q1 = trapz(q_temp1)*ts;
        q2 = trapz(q_temp2)*ts;
        q = [q1;q2];

    %   Collect: Joint acceleration:
        aq = [aq, qddr];

    %   Conversion to task space again:
        xcdd = calcXdd([q',qd',qddr'])';
        xcd = calcXd([q',qd'])';
        xc = calcForward(q')';   

        xdd_out = [xdd_out, xcdd'];
        xd_out = [xd_out, xcd'];
        x_out = [x_out, xc'];
    end

    close all
    % m-files of scripts plotting the output:

    % Task space plots
    run('figures_plot_xin_xout.m');
    run('figures_plot_taskspace_traj.m');
    run('figures_plot_taskspace_traj_robot.m');
    run('figures_plot_force_xdd');

    % Joint space plots
    run('figures_plot_joint_acceleration_in_out.m');
    run('figures_plot_joint_variables.m');

%     close all
    run('save_sessions')
%     close all
end




%%
X1 = [time_span', X1]
X1dot = [time_span', X1dot]
X1ddot = [time_span', X1ddot]

%%




aq = []
% time = linspace(0,10,60);
% qc = [[linspace(0,deg2rad(30),30) linspace(deg2rad(30),deg2rad(30),30)]', [linspace(0,deg2rad(30),30), linspace(deg2rad(30),deg2rad(30),30)]'];
% qcdot = [gradient(qc')'];
% qcddot = [gradient(qcdot')'];

ts = 0.001;
time_span = 0:ts:2;
qc = [pi/3*sin(2*pi*1*time_span)' pi/3*sin(2*pi*1*time_span)'];
qcdot = gradient(qc', ts)';
qcddot = gradient(qcdot', ts)';

% torque = robot2.rne(qc,qcdot,qcddot);
% qddr = robot2.accel([qc(:,1) qc(:,2)], [qcdot(:,1) qcdot(:,2)] ,[torque(:,1) torque(:,2)])



X1 = [pi/3*sin(2*pi*1*time_span)' pi/3*sin(2*pi*1*time_span)' zeros(1,length(time_span))']
X1dot = gradient(X1', ts)';
X1ddot = gradient(X1dot', ts)';

X1 = [time_span', X1]
X1dot = [time_span', X1dot];
X1ddot = [time_span', X1ddot];


% qc = [time_span', qc];
% qcdot = [time_span', qcdot];
% qcddot = [time_span', qcddot];

omega_n = 2;
KP = eye(2)*omega_n^2
KV = eye(2)*(2*omega_n)
KI = eye(2)* 1

% For simulink



%%


Qcol = []
Mcol = []
qd = []
% aq = [aq zeros(2,1)]
aq = []
% qd = [qd zeros(2,1)]
qd = zeros(2,length(qc));
q = zeros(2,length(qc));

qddr = [];

for i = 2:length(qc)
%     j = i-1;
    q = qc(i,:);
%     qcj = qc(j,:);
    qdot = qcdot(i,:);
%     qcdotj = qcdot(j,:);
    qddot = qcddot(i,:);
%     qcddotj = qcddot(i,:);
    u1 = [q qdot qddot];

    
%     Torque computation:
    Q = RNE_sim2(u1);    
    
    
    u2 = [Q' q qdot];
    
    Qcol = [Qcol Q];
    [aq_r M] = forwardDyn(u2);
    qddrt = robot2.accel(q,qdot,Q');
%     Mcol(:,:,i-1) = Mtest(:,:,1);
    Mcol(:,:,i) = M;
%     Mcol = [Mcol Mtest];
    qddr = [qddr,qddrt];
    aq = [aq, aq_r];
    
%     qdt = aq(:,i)'+aq(:,j)' * ts;    
%     qd = [qd, qdt'];
  
%     qd = integral(aq',aq(:,j)',aq(:,i)');
%     q = [q, qt];
    
    
    
    
end

close all
torque = robot2.rne(qc,qcdot,qcddot);
% qddr = robot2.accel([qc(:,1) qc(:,2)], [qcdot(:,1) qcdot(:,2)] ,[torque(:,1) torque(:,2)])
% qd = trapz(time_span,aq)
% figure
% plot(qd)




% figure()
% plot(q')
% grid on
% hold on
% plot(qc)


Mt = robot2.inertia([qc]);
% 
figure()
subplot(1,2,1),plot(Mcol(1,:))
hold on;
subplot(1,2,1),plot(Mcol(2,:))
title('Computed M');
xlabel('elements in matrix');
ylabel('[Nm]');
grid on;
hold off
% 
subplot(1,2,2),plot(Mt(1,:)')
hold on;
subplot(1,2,2),plot(Mt(2,:)')
title('Toolbox M');
xlabel('elements in matrix');
ylabel('[Nm]');
grid on;
hold off
% 
% 
% figure()
% subplot(1,2,1),plot(Qcol')
% subplot(1,2,2),plot(torque)
% 
% 
% 
% 
figure()

subplot(1,2,1),plot(aq')
title('computed acceleration')
grid on
subplot(1,2,2),plot(qddr')

figure()
plot(qcdot)
title('desired acceleration')


%%
figure()
plot2()




% torque = robot2.rne(qci,qcdoti,qcddot);
%%
J=[];
for i = i:length(time_span)
    qci = qc(i,:)
    Ji = calcJacobian(qci);
    J(:,:,i) = Ji;
end
%%
aq = [zeros(2,1) aq]'

%%

% Force control testing

% xdd = ax + Wq * (Fe - af)
% ax = Jq * aq + Jd * qd
% Wq = J*inv(M)*J'
% 
% 
% ax = xdd_d - inv(M)
% i = 100
% Jnew = J(:,:,i)
% Jprev = J(:,:,i-1)

% % test = [Jprev Jnew];
% testres = diff(J(:,:,i))
% 

% Jd = Jnew - Jprev
% Ji = Jnew;


% ax = []
% for i = 1:length(time_span)
%     [Jxx,Jyy]=gradient(J(:,:,i));
%     ax = [ax J(:,:,i)*aq(i,:)' + Jyy*qcdot(i,:)'];
%     
% end





% Notes to self, issues with gradient and numerical derrivatives
% discussions on interpolation techniqe

% Analytical Jacobian
% *above is ignored for now (Analytical Jacobian is ignored)



% % numerical differences
% Jd = []
% for i = 1:length(J)
%     for k = 1:size(J,1)-1
%         Jd(k,:,i) = J(k+1,:,i) - J(k,:,i);
%     end
% end


%%
% % i = 10001;
% % Jd = gradient(J(:,:,i))
% % ax = Jq * aq + Jd * qcdot
% % Wq = J*inv(M)*J'
% 
% 
% m1 = 5
% m2 = 5
% 
% omega_n = 10
% zeta = 1
% B = 2*eye(2)*zeta*omega_n
% K = eye(2)*omega_n^2
% M = eye(2)*[m1;m2]
% 
% 
% 
%%

% Testing Impedance Control


























