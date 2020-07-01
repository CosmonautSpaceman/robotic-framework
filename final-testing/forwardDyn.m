function [qacc, M] = forwardDyn(u)
Q = [u(1) u(2)];
qc = [u(3) u(4)];
% qc_prev = [u(5) u(6)];

qcdot = [u(5) u(6)];
% qcdot_prev = [u(9) u(10)];

Q = [Q(1) Q(2)]';
M = calc_M_sim(qc,qcdot);
CG = calc_CG_sim(qc,qcdot);
qacc = inv(M)*(Q - CG);
end