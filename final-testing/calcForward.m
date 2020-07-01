function [translation] = calcForward(u)
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
translation = T(1:3,4);
end