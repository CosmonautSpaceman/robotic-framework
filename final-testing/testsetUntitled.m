clear all
close all
clc

% Inertia Computation

% syms q1 q2 real
q1 = 0.35;
q2 = 0.35;

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

q1c = q1;
q2c = q2;

C = [                                                                                                                                                                                                                                                                                                                                                    (L3^2*lc2^2*m3*sin(2*real(q1c)))/6 - (lc1*lc2*m2*sin(q1c + q2c - conj(q1c)))/2 - (lc1*lc2*m2*sin(conj(q1c) - q1c + conj(q2c)))/2 - (lc2^2*m3*r3^2*sin(2*real(q1c)))/2 - (L3^2*l1*lc2*m3*sin(q1c + q2c - conj(q1c) - 2*conj(q2c)))/12 + (L3^2*lc1*lc2*m3*sin(q1c + q2c - conj(q1c) - 2*conj(q2c)))/12 - (L3^2*l1*lc2*m3*sin(2*real(q1c) - conj(q2c)))/12 + (L3^2*lc1*lc2*m3*sin(2*real(q1c) - conj(q2c)))/12 - (3*l1*lc2*m3*r3^2*sin(q1c + q2c - conj(q1c) - 2*conj(q2c)))/4 + (3*lc1*lc2*m3*r3^2*sin(q1c + q2c - conj(q1c) - 2*conj(q2c)))/4 + (l1*lc2*m3*r3^2*sin(2*real(q1c) - conj(q2c)))/4 - (lc1*lc2*m3*r3^2*sin(2*real(q1c) - conj(q2c)))/4 + (L3^2*l1*lc2*m3*sin(q2c - 2*real(q1c)))/12 - (L3^2*lc1*lc2*m3*sin(q2c - 2*real(q1c)))/12 + (L3^2*l1*lc2*m3*sin(q1c + 2*q2c - conj(q1c) - conj(q2c)))/12 - (L3^2*lc1*lc2*m3*sin(q1c + 2*q2c - conj(q1c) - conj(q2c)))/12 - (l1*lc2*m3*r3^2*sin(q2c - 2*real(q1c)))/4 + (lc1*lc2*m3*r3^2*sin(q2c - 2*real(q1c)))/4 + (3*l1*lc2*m3*r3^2*sin(q1c + 2*q2c - conj(q1c) - conj(q2c)))/4 - (3*lc1*lc2*m3*r3^2*sin(q1c + 2*q2c - conj(q1c) - conj(q2c)))/4, (9*l1^2*m3*r3^2*sin(2*real(q1c) - 2*real(q2c)))/8 - (3*L3^2*lc1^2*m3*sin(2*real(q1c) - 2*real(q2c)))/8 - (3*L3^2*l1^2*m3*sin(2*real(q1c) - 2*real(q2c)))/8 + (9*lc1^2*m3*r3^2*sin(2*real(q1c) - 2*real(q2c)))/8 - (3*lc1*lc2*m2*sin(q1c + q2c - conj(q1c)))/2 - (L3^2*lc2^2*m3*sin(2*real(q1c)))/24 + l1*lc2*m2*sin(conj(q1c) - q1c + conj(q2c)) - (3*lc1*lc2*m2*sin(conj(q1c) - q1c + conj(q2c)))/2 + (lc2^2*m3*r3^2*sin(2*real(q1c)))/8 + (3*L3^2*l1*lc1*m3*sin(2*real(q1c) - 2*real(q2c)))/4 - (L3^2*l1*lc2*m3*sin(q1c + q2c - conj(q1c) - 2*conj(q2c)))/6 + (L3^2*lc1*lc2*m3*sin(q1c + q2c - conj(q1c) - 2*conj(q2c)))/6 + (5*L3^2*l1*lc2*m3*sin(2*real(q1c) - conj(q2c)))/24 - (5*L3^2*lc1*lc2*m3*sin(2*real(q1c) - conj(q2c)))/24 - (9*l1*lc1*m3*r3^2*sin(2*real(q1c) - 2*real(q2c)))/4 - (3*l1*lc2*m3*r3^2*sin(q1c + q2c - conj(q1c) - 2*conj(q2c)))/2 + (3*lc1*lc2*m3*r3^2*sin(q1c + q2c - conj(q1c) - 2*conj(q2c)))/2 - (5*l1*lc2*m3*r3^2*sin(2*real(q1c) - conj(q2c)))/8 + (5*lc1*lc2*m3*r3^2*sin(2*real(q1c) - conj(q2c)))/8 - (5*L3^2*l1*lc2*m3*sin(q2c - 2*real(q1c)))/24 + (5*L3^2*lc1*lc2*m3*sin(q2c - 2*real(q1c)))/24 + (L3^2*l1*lc2*m3*sin(q1c + 2*q2c - conj(q1c) - conj(q2c)))/6 - (L3^2*lc1*lc2*m3*sin(q1c + 2*q2c - conj(q1c) - conj(q2c)))/6 + (5*l1*lc2*m3*r3^2*sin(q2c - 2*real(q1c)))/8 - (5*lc1*lc2*m3*r3^2*sin(q2c - 2*real(q1c)))/8 + (3*l1*lc2*m3*r3^2*sin(q1c + 2*q2c - conj(q1c) - conj(q2c)))/2 - (3*lc1*lc2*m3*r3^2*sin(q1c + 2*q2c - conj(q1c) - conj(q2c)))/2;
 (3*L3^2*l1^2*m3*sin(2*real(q1c) - 2*real(q2c)))/8 + (3*L3^2*lc1^2*m3*sin(2*real(q1c) - 2*real(q2c)))/8 - (9*l1^2*m3*r3^2*sin(2*real(q1c) - 2*real(q2c)))/8 - (9*lc1^2*m3*r3^2*sin(2*real(q1c) - 2*real(q2c)))/8 + (lc1*lc2*m2*sin(q1c + q2c - conj(q1c)))/2 + (5*L3^2*lc2^2*m3*sin(2*real(q1c)))/24 + (lc1*lc2*m2*sin(conj(q1c) - q1c + conj(q2c)))/2 - (5*lc2^2*m3*r3^2*sin(2*real(q1c)))/8 - (3*L3^2*l1*lc1*m3*sin(2*real(q1c) - 2*real(q2c)))/4 + (L3^2*l1*lc2*m3*sin(q1c + q2c - conj(q1c) - 2*conj(q2c)))/12 - (L3^2*lc1*lc2*m3*sin(q1c + q2c - conj(q1c) - 2*conj(q2c)))/12 - (7*L3^2*l1*lc2*m3*sin(2*real(q1c) - conj(q2c)))/24 + (7*L3^2*lc1*lc2*m3*sin(2*real(q1c) - conj(q2c)))/24 + (9*l1*lc1*m3*r3^2*sin(2*real(q1c) - 2*real(q2c)))/4 + (3*l1*lc2*m3*r3^2*sin(q1c + q2c - conj(q1c) - 2*conj(q2c)))/4 - (3*lc1*lc2*m3*r3^2*sin(q1c + q2c - conj(q1c) - 2*conj(q2c)))/4 + (7*l1*lc2*m3*r3^2*sin(2*real(q1c) - conj(q2c)))/8 - (7*lc1*lc2*m3*r3^2*sin(2*real(q1c) - conj(q2c)))/8 + (7*L3^2*l1*lc2*m3*sin(q2c - 2*real(q1c)))/24 - (7*L3^2*lc1*lc2*m3*sin(q2c - 2*real(q1c)))/24 - (L3^2*l1*lc2*m3*sin(q1c + 2*q2c - conj(q1c) - conj(q2c)))/12 + (L3^2*lc1*lc2*m3*sin(q1c + 2*q2c - conj(q1c) - conj(q2c)))/12 - (7*l1*lc2*m3*r3^2*sin(q2c - 2*real(q1c)))/8 + (7*lc1*lc2*m3*r3^2*sin(q2c - 2*real(q1c)))/8 - (3*l1*lc2*m3*r3^2*sin(q1c + 2*q2c - conj(q1c) - conj(q2c)))/4 + (3*lc1*lc2*m3*r3^2*sin(q1c + 2*q2c - conj(q1c) - conj(q2c)))/4,                                                                                                                                                                                                                                                                                        (l1*lc2*m2*sin(q1c + q2c - conj(q1c)))/2 - (lc1*lc2*m2*sin(q1c + q2c - conj(q1c)))/2 + (L3^2*lc2^2*m3*sin(2*real(q1c)))/24 + (l1*lc2*m2*sin(conj(q1c) - q1c + conj(q2c)))/2 - (lc1*lc2*m2*sin(conj(q1c) - q1c + conj(q2c)))/2 - (lc2^2*m3*r3^2*sin(2*real(q1c)))/8 - (L3^2*l1*lc2*m3*sin(q1c + q2c - conj(q1c) - 2*conj(q2c)))/48 + (L3^2*lc1*lc2*m3*sin(q1c + q2c - conj(q1c) - 2*conj(q2c)))/48 - (L3^2*l1*lc2*m3*sin(2*real(q1c) - conj(q2c)))/48 + (L3^2*lc1*lc2*m3*sin(2*real(q1c) - conj(q2c)))/48 - (3*l1*lc2*m3*r3^2*sin(q1c + q2c - conj(q1c) - 2*conj(q2c)))/16 + (3*lc1*lc2*m3*r3^2*sin(q1c + q2c - conj(q1c) - 2*conj(q2c)))/16 + (l1*lc2*m3*r3^2*sin(2*real(q1c) - conj(q2c)))/16 - (lc1*lc2*m3*r3^2*sin(2*real(q1c) - conj(q2c)))/16 + (L3^2*l1*lc2*m3*sin(q2c - 2*real(q1c)))/48 - (L3^2*lc1*lc2*m3*sin(q2c - 2*real(q1c)))/48 + (L3^2*l1*lc2*m3*sin(q1c + 2*q2c - conj(q1c) - conj(q2c)))/48 - (L3^2*lc1*lc2*m3*sin(q1c + 2*q2c - conj(q1c) - conj(q2c)))/48 - (l1*lc2*m3*r3^2*sin(q2c - 2*real(q1c)))/16 + (lc1*lc2*m3*r3^2*sin(q2c - 2*real(q1c)))/16 + (3*l1*lc2*m3*r3^2*sin(q1c + 2*q2c - conj(q1c) - conj(q2c)))/16 - (3*lc1*lc2*m3*r3^2*sin(q1c + 2*q2c - conj(q1c) - conj(q2c)))/16]

g_load = [0;0;0]





% I1m = max(max(I1))
% I2m = max(max(I2))
% I3m = max(max(I3))
% I4m = max(max(I4))
% 
% I_imax = [I1m; I2m; I3m; I4m]
% D_hat = diag(I_imax)

% Jmi =  1340*10^(-7);
% gear = 53;
% Jm = eye(2) * ((gear^2) * Jmi);



%%

% P1 = m1*g*d_1;
% P2 = m2*g*d_1;
% 
% P = P1 + P2;
% 
% phi_1 = functionalDerivative(P,q1);
% phi_2 = functionalDerivative(P,q2);
% 
% g_load = [phi_1;phi_2];
