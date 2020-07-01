function aq = cartPD(u)
% 7 signals
% desired signals (three)
x_in = [u(1:3)];
xd_in = [u(4:6)];
ax1 = [u(7:9)];

% feedback signals (two)
xc = [u(10:12)];
xcd = [u(13:15)];

% joint signals (two)
q = [u(16:17)];
qd = [u(18:19)];

% Control parameters
KdIn = [eye(2).*((0.01)^2) zeros(2,1); zeros(1,3)];
KpIn = [eye(2).*(0.01)*2 zeros(2,1); zeros(1,3)];

% error calculations
e = x_in - xc;
ed = xd_in - xcd;

% computed acceleration:
w = ax1 + KdIn*ed + KpIn*e;

% Jacobian calculations
Jdq = calcJacobianDot([q(1), q(2), qd(1), qd(2)]) * qd;
J = calcJacobian(q);
J = J(1:2,1:2);

lambda = 0.5;
Jinv = (J' * inv(J*J'+eye(2)*lambda))';

aq = Jinv * (w(1:2) - Jdq(1:2));
end