function y = calcTorqueForce(u)
tau = [u(1) u(2)]';
qc = [u(3) u(4)];
J = calcJacobian(qc);
% J = J'
% J*J'

lambda = 0.1;
Jinv = (J' * inv(J*J'+eye(6)*lambda))';
TestSingularity = det(J(1:2,1:2));


if (TestSingularity < 1e-11) & (TestSingularity > -1e-11)
    y = [0;0;0];
else
    y = Jinv' * [tau; 0; 0; 0; 0];
    y = [y; 0];
    if(isnan(y))
        y = [0;0;0];
    end
end
end