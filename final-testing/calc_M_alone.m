function Lx = calc_M_alone(u)
xdd =[u(1) u(2) u(3)]';

% qc = [u(4) u(5) u(6) u(7)];
% qcdot = [u(8) u(9) u(10) u(11)];
% 
% M = calc_M_sim(qc,qcdot)
% J = calcJacobian(qc(1:2))
% Lambda = (J*inv(M)*J')'

m1 = 5; m2=5; m3 = 0;


KM = eye(3).*[m1;m2;m3];
Lx=KM*xdd;

% operational space inertia matrix with spatial vectors
% Lx = Lambda(1:length(xdd),1:length(xdd))*xdd;

end