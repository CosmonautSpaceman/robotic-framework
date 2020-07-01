function y = fdynLagrange(u)
Q = [u(1) u(2)];
qc = [u(3) u(4)]
% qc_prev = [u(5) u(6)];

qcdot = [u(5) u(6)]
% qcdot_prev = [u(9) u(10)];

Q = [Q(1) Q(2)]';
M = lagrangeInertia(qc);
C = lagrangeCoriolis([qc,qcdot]);
G = [0;0];

M

CG = (C-G)
Minv = inv(M)
qacc = [inv(M)*(Q - C-G)]
% res = M;



y = qacc;
