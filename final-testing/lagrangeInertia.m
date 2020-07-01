function M = lagrangeInertia(u)

q1 = u(1);
q2 = u(2);

m2 = 5; r2 = 0.05; l1 = 0.7; L2 = l1;
m3 = 5; r3 = 0.05; l2 = 0.7; L3 = l2;

lc1 = l1/2; lc2 = l2/2; m1 = m2;

J1c_v = [ -lc1*sin(q1), 0;
          lc1*cos(q1), 0;
                    0, 0];

J1c_w = [ 0, 0;
         0, 0;
         1, 0];

J2c_v = [ - lc1*sin(q1) - lc2*cos(q1)*sin(q2) - lc2*cos(q2)*sin(q1), l1*sin(q1) - lc1*sin(q1) - lc2*cos(q1)*sin(q2) - lc2*cos(q2)*sin(q1);
           lc1*cos(q1) + lc2*cos(q1)*cos(q2) - lc2*sin(q1)*sin(q2), lc1*cos(q1) - l1*cos(q1) + lc2*cos(q1)*cos(q2) - lc2*sin(q1)*sin(q2);
                                                                 0,                                                                    0];

J2c_w = [ 2*l1*sin(q1) - 2*lc1*sin(q1) - 2*lc2*cos(q1)*sin(q2) - 2*lc2*cos(q2)*sin(q1), l1*sin(q1) - lc1*sin(q1) - lc2*cos(q1)*sin(q2) - lc2*cos(q2)*sin(q1);
         2*lc1*cos(q1) - 2*l1*cos(q1) + 2*lc2*cos(q1)*cos(q2) - 2*lc2*sin(q1)*sin(q2), lc1*cos(q1) - l1*cos(q1) + lc2*cos(q1)*cos(q2) - lc2*sin(q1)*sin(q2);
                                                                                    0,                                                                    0];
R1 = [ cos(q1), -sin(q1), 0;
 sin(q1),  cos(q1), 0;
       0,        0, 1];
 
R2 =[ cos(q2), -sin(q2), 0;
 sin(q2),  cos(q2), 0;
       0,        0, 1];


% I_1 = (1/12)*m1*(3*r1^2 + L1^2);
% I_1yy = (1/2)*m1*r1^2;

I_2 = (1/12)*m2*(3*r2^2 + L2^2);
I_2yy = (1/2)*m2*r2^2;

I_3 = (1/12)*m3*(3*r3^2 + L3^2);
I_3yy = (1/2)*m3*r3^2;

% I_4 = (1/12)*m4*(3*r4^2 + L4^2);
% I_4yy = (1/2)*m4*r4^2;

I2 = [I_2 0 0; 0 I_2yy 0; 0 0 I_2];
I3 = [I_3 0 0; 0 I_3yy 0; 0 0 I_3];
% I4 = [I_4 0 0; 0 I_4yy 0; 0 0 I_4];
I_col = (J1c_w' * R1 * I2 * R1' * J1c_w + J2c_w' * R2 * I3 * R2' * J2c_w)
D = (m1*J1c_v' * J1c_v + m2*J2c_v' * J2c_v) + I_col

Jmi =  1340*10^(-7);
gear = 53;
Jm = eye(2) * ((gear^2) * Jmi);

M = D + Jm;

end