function F = impedanceForceControl(u)
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


robot2 = SerialLink( [ Revolute('a', 0.7,'m',5,'r',[-0.35; 0; 0], 'I', I1) ...
    Revolute('a', 0.7,'m',5,'r',[-0.35; 0; 0],  'I', I2)],'name', 'my robot');
robot2.gravity = [0; 0; -9.8];


x = [u(1), u(2), u(3)]';
xd = [u(4), u(5), u(6)]';
xdd = [u(7), u(8), u(9)]';

xc = [u(10), u(11), u(12)]';
xcd = [u(13), u(14), u(15)]';
xcdd = [u(16), u(17), u(18)]';

% qc = [u(19), u(20)];

% Md = robot2.cinertia(qc);

% Jtest = robot2.jacob0(qc);
% TestSingularity = det(Jtest(1:2,1:2));
% 
% if (TestSingularity ~= 0)
%     Jtest = Jtest(1:2,1:2);
%     Jtestinv = inv(Jtest);
%     Mtest = robot2.inertia(qc);
%     Md = Jtestinv'*Mtest*Jtestinv;
%     Md = [Md zeros(2,1); zeros(1,3)]
% else
%     Md = zeros(3,3);
% %     Md = Jtestinv'*Mtest*Jtestinv;
% %     Md = [Md zeros(2,1); zeros(1,3)]
% end

% 5kg and state space normalization of 70
Md = eye(3).*(5/70);

% ts =  0.0101;

% Kd = (eye(3).*10^2);
% Bd = eye(3).*(2*10);
Kd = (eye(3));
Bd = eye(3).*0.2429;

F = (Md * (xdd-xcdd)) + (Bd * (xd-xcd) ) + (Kd * (x- xc));
end