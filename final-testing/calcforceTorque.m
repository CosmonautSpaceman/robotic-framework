function Fe = calcforceTorque(u)
xd = [u(1) u(2) u(3) 0 0 0]';
qc = [u(4) u(5)];
J = calcJacobian(qc);
TestSingularity = det(J(1:2,1:2));

if (TestSingularity < 1e-3) & (TestSingularity > -1e-3)
    Fe = [0;0];
else
    Fe = J' * xd;
    if(isnan(Fe))
        Fe = [0;0];
    end
end
end