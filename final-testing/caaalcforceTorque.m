function Tau = calcforceTorque(u)
Fe = [u(1) u(2) u(3) 0 0 0]';
qc = [u(4) u(5)];
J = calcJacobian(qc);
TestSingularity = det(J(1:2,1:2));

if (TestSingularity < 1e-3) & (TestSingularity > -1e-3)
    Tau = [0;0];
else
    Tau = J' * Fe;
    if(isnan(Tau))
        Tau = [0;0];
    end
end
end