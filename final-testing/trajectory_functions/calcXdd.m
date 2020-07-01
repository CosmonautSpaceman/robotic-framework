function ax = calcXdd(u)

% Robot Link variables
% % m1 = 5; r1 = 0.1; L1 = 0.15;
% m2 = 5; r2 = 0.05; L2 = 0.7;
% m3 = 5; r3 = 0.05; L3 = 0.7;
% % m4 = 4; r4 = 0.03; L4 = 0.4;
% 
% % Inertia Matrix
% % I_1 = (1/12)*m1*(3*r1^2 + L1^2);
% % I_1yy = (1/2)*m1*r1^2;
% 
% I_2 = (1/12)*m2*(3*r2^2 + L2^2);
% I_2yy = (1/2)*m2*r2^2;
% 
% I_3 = (1/12)*m3*(3*r3^2 + L3^2);
% I_3yy = (1/2)*m3*r3^2;
% % 
% % I_4 = (1/12)*m4*(3*r4^2 + L4^2);
% % I_4yy = (1/2)*m4*r4^2;
% 
% 
% % Remember the order of rigid bodies
% I1 = [I_2 0 0; 0 I_2yy 0; 0 0 I_2];
% I2 = [I_3 0 0; 0 I_3yy 0; 0 0 I_3];
% 
% 
% robot2 = SerialLink( [ Revolute('a', 0.7,'m',5,'r',[-0.35; 0; 0], 'I', I1) ...
%     Revolute('a', 0.7,'m',5,'r',[-0.35; 0; 0],  'I', I2)],'name', 'my robot');
% robot2.gravity = [0; 0; -9.8];


qc = [u(1) u(2)]';
qcdot = [u(3) u(4)]';
qcddot = [u(5) u(6)]';
qin = [qc qcdot];
aq = qcddot;

J = calcJacobian(qc);
% J = robot2.jacob0(qc);
% Jd = robot2.jacob_dot(qc,qcdot);
Jd = calcJacobianDot(qin);
Jd = Jd*qcdot;

% testSingularity = det(J);
% disp('compute-joint to task');
TestSingularity = det(J(1:2,1:2));
rank(J);

if (TestSingularity < 1e-11) & (TestSingularity > -1e-11)
% if (TestSingularity == 0)
    ax = [0;0;0];
else
%     ax = J(1:3,1:2) * aq + Jd(1:3,1:2) * qcdot;
    ax = J(1:3,1:2) * aq + Jd(1:3);
    ax = ax(1:3,:);
end
% qc = [u(1) u(2) u(3) u(4)];
% qcdot = [u(5) u(6) u(7) u(8)];
% qcddot = [u(9) u(10) u(11) u(12)];

end