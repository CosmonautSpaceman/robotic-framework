function J = calcJacobian(u)

alpha =[0 0];
d = [0 0];
theta = [u(1) u(2)];
a = [0.7 0.7];
collect = [];
T = eye(4);

for i = 1:2
    collect{i} = [cos(theta(i)) -cos(alpha(i))*sin(theta(i)) sin(alpha(i))*sin(theta(i)) a(i)*cos(theta(i));
        sin(theta(i)) cos(alpha(i))*cos(theta(i)) -sin(alpha(i))*cos(theta(i)) a(i)*sin(theta(i));
        0 sin(alpha(i)) cos(alpha(i)) d(i);
        0 0 0 1];   
    T = T* collect{i};
end
A1 = collect{1};
A2 = collect{2};

% syms l1 l2
% A1c = subs(A1, [a1,a2],[l1,l2]);

% R1 = A1(1:3,1:3);
% R2 = A2(1:3,1:3);


J1 = [cross([0 0 1]',T(1:3,4)); [0 0 1]'];
J2 = [cross(A1(1:3,3),T(1:3,4)-A1(1:3,4)) ; A1(1:3,3)];
J = [J1 J2];
end