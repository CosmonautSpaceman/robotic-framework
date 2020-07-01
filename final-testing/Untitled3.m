clear all
close all
clc

n_i = 53;
J_Mi = 1340*10^(-7);
k_T = 0.24;
f_eff = 2.6*10^(-5);
g = 9.8;

% act_mod = (k_Ti * u_i) - (1/n_i * tau_i) == J_Mi * q_dd_Mi + f_Mi*q_d_Mi;
% q_dd_Mi= ((k_Ti * u_i) - (1/n_i * tau_i)  - f_Mi*q_d_Mi) / J_Mi;



% run('actuatorTest.m')
run('IndependentJoint.m')
clc
% Here we add the following:

syms K_P K_D s




G_s = (1/n^2)
N_s = Jeff_c*s^2 + f_eff*s + k_T*K_P + k_T*K_D*s
F_s = (K_D * s * k_T + K_P* k_T)

q1_test = (F_s * G_s)/ N_s(1)
% q2_test = (F_s * G_s)/ N_s(2)


syms omega_n zeta

% omega_n =  655.03890785716153861254900562852
% omega_n =  0.044232133946276766237923710560971
% omega_n = 0.011708062814524971573368792455476
omega_n = 15
zeta = 1
% qr_i = pi
qr_i = 0.35


K_P = []
K_D = []
K_I = []

% for i = 1:2
% 	K_P(i) = (omega_n^2 * Jeff_c(i))/k_T
% 	K_D(i) = ((2*omega_n*zeta*Jeff_c(i))/(k_T)-(f_eff)/(k_T))
% end


for i = 1:2
	K_P(i) = ((omega_n^2) * Jeff_c(i) + 2 * zeta * omega_n)/k_T
	K_D(i) = ( 1 + (2 * omega_n * zeta * Jeff_c(i)) - (f_eff) )/(k_T)
    K_I(i) = (omega_n^2)/k_T
end


Tlmax = max(Tl_c');
Tl = double(Tlmax(1))
Jeff = double(Jeff_c(1))
KP1 = double(K_P(1))
KD1 = double(K_D(1))
KI1 = double(K_I(1))

KP2 = double(K_P(2))
KD2 = double(K_D(2))
KI2 = double(K_I(2))
