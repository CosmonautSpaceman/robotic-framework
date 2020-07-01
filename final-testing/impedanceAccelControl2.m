function ax = impedanceAccelControl2(u,Md,Bd,Kd)
% Kd = (eye(3).*3);
% Bd = eye(3).*5;

x_t = [u(1) u(2) u(3)]';
xd_t = [u(4) u(5) u(6)]';
xdd_d = [u(7) u(8) u(9)]';
F = [u(10) u(11) u(12)]';
% qc = [u(13), u(14)];

xc = [u(13), u(14), u(15)]';
xcd = [u(16), u(17), u(18)]';
% xcdd = [u(19), u(20), u(21)]';


% qc = [u(22),u(23)];

% Md = robot2.cinertia(qc);
% Jtest = robot2.jacob0(qc);
% TestSingularity = det(Jtest(1:2,1:2));

% if (TestSingularity ~= 0)
%     Jtest = Jtest(1:2,1:2);
%     Jtestinv = inv(Jtest);
%     Mtest = robot2.inertia(qc);
%     Md = Jtestinv'*Mtest*Jtestinv;
% %     Md = [Md zeros(2,1); zeros(1,3)]
% else
%     Md = zeros(2,2);
% %     Md = Jtestinv'*Mtest*Jtestinv;
% %     Md = [Md zeros(2,1); zeros(1,3)]
% end



% Md = inv(eye(2).*(5));


% Md2 = [Md 0; zeros(1,3)]

% l1 = (Bd*xd_t + Kd*x_t + F)

ax = xdd_d - ([Md zeros(2,1); zeros(1,3)]*(Bd*(xcd-xd_t) + Kd*(xc-x_t) + F));

end