function qd = calcQd(u)

tx = u(1);
ty = u(2);
tz = u(3);

X = [tx,ty,tz]';
qc = [u(4),u(5)]';


% 
% % Robot Link variables
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



J = calcJacobian(qc);
% J = robot2.jacob0(qc);
% Jinv = inv(J(1:2,1:2));
lambda = 0.5;
Jinv = (J' * inv(J*J'+eye(6)*lambda))';
TestSingularity = det(J(1:2,1:2));

if (TestSingularity < 1e-11) & (TestSingularity > -1e-11)
% if (TestSingularity == 0)
% if (TestSingularity == 0)
    qd = [0;0];
else
    qd = Jinv(1:2,1:2)*X(1:2);
%     if (qd(1) > 1000) | (qd(1) < -1000)
%         qd = [0;0];
%     end
    
%     aq = [0;0];
end


% q1 = atan2(- tx^2 - ty^2, -(- tx^4 - 2*tx^2*ty^2 + (49*tx^2)/25 - ty^4 + (49*ty^2)/25)^(1/2)) - atan2(-tx, -ty);
% q2 = - atan2(-tx, -ty) + atan2(- tx^2 - ty^2, (- tx^4 - 2*tx^2*ty^2 + (49*tx^2)/25 - ty^4 + (49*ty^2)/25)^(1/2));


end