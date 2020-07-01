clear all
close all
clc

T1 = [0 0 0 0;0 0 0 -0.5; -1 0 0 0; 0 0 0 1]; 
T2 = [0 0 0 0.35;0 0 0 0.5; -1 0 0 0; 0 0 0 1]; 
T3 = [0 0 0 0.7;0 0 0 -0.5; -1 0 0 0; 0 0 0 1]; 
T4 = [0 0 0 1.05 ;0 0 0 0.5; -1 0 0 0; 0 0 0 1]; 
T5 = [0 0 0 1.4 ;0 0 0 -0.5; -1 0 0 0; 0 0 0 1]; 

Ttask = [1 0 0 1.5;0 1 0 0; 0 0 1 0.2; 0 0 0 1]; 

a_4 = 1.10; 
d_1 = 1.6; 

TN = {T1 T2 T3 T4 T5}; collect = []; tp_cell = []

for i = 1:5
    collect = Ttask*TN{1,i};
    tp_cell{i} = collect;
end

q1 = []; q2 = []; q3 = []; q4 = []; tp = []

for i = 1:5
    tp = tp_cell{i};
    q1(i) =  atan2( tp(2,4), tp(1,4));
    q2(i) =  atan2( sin(q1(i))*(tp(2,4)-a_4*tp(2,1))+cos(q1(i))*(tp(1,4)-a_4*tp(1,1)),tp(3,4)-d_1-a_4*tp(3,1));
    q3(i) = (sin(q1(i))*(tp(2,4)-a_4*tp(2,1))+cos(q1(i))*(tp(1,4)-a_4*tp(1,1)))/sin(q2(i));
    q4(i) = atan2( tp(3,1), ((-sin(q1(i))* tp(2,4) - cos(q1(i)) * tp(1,4) + q3(i) * sin(q2(i)))/a_4) ) - q2(i)+2*pi;
end

T = [q1;q2;q3;q4]

% Duration is 2 seconds
t0 = 0;
tf = 2;

syms q0; 
v0 = 0; 
a0 = 0; 
syms  qf; 
vf = 0; 
af = 0; 

a_mat_1 = inv([1 t0 t0^2 t0^3 t0^4 t0^5; 
               0 1 2*t0 3*t0^2 4*t0^3 5*t0^4; 
               0 0 2 6*t0 12*t0^2 20*t0^3; 
               1 tf tf^2 tf^3 tf^4 tf^5;
               0 1 2*tf 3*tf^2 4*tf^3 5*tf^4; 
               0 0 2 6*tf 12*tf^2 20*tf^3]); 

traj_vec = [q0; v0; a0; qf; vf; af]; 

segments = []; segments_collect = []; 

for i = 1:4
    q1seg = double(subs(a_mat_1 * traj_vec, [q0,qf], [q1(i), q1(i+1)]));
    q2seg = double(subs(a_mat_1 * traj_vec, [q0,qf], [q2(i), q2(i+1)]));
    q3seg = double(subs(a_mat_1 * traj_vec, [q0,qf], [q3(i), q3(i+1)]));
    q4seg = double(subs(a_mat_1 * traj_vec, [q0,qf], [q4(i), q4(i+1)]));

    segments = [q1seg'; q2seg'; q3seg'; q4seg']; 
    segments_collect{i} = segments; 
end

seg1 = [];seg2 = [];seg3 = [];seg4 = [];

seg1 = segments_collect{1}
seg2 = segments_collect{2}
seg3 = segments_collect{3}
seg4 = segments_collect{4}

% global GM 

GM = [seg1;seg2;seg3;seg4]

%%
% duration 2 seconds
t = linspace(0,2);
% t2 = linspace(2,4);
% t3 = linspace(4,6);
% t4 = linspace(6,8);

a_s1 = seg1
a_s2 = seg2
a_s3 = seg3
a_s4 = seg4

qtraj1 = a_s1(1,6) .* t.^5 + a_s1(1,5) .*t.^4 + a_s1(1,4) .*t.^3 + a_s1(1,3) .*t.^2 + a_s1(1,2) .*t + a_s1(1,1);
qtraj2 = a_s2(1,6) .* t.^5 + a_s2(1,5) .*t.^4 + a_s2(1,4) .*t.^3 + a_s2(1,3) .*t.^2 + a_s2(1,2) .*t + a_s1(1,1);
qtraj3 = a_s3(1,6) .* t.^5 + a_s3(1,5) .*t.^4 + a_s3(1,4) .*t.^3 + a_s3(1,3) .*t.^2 + a_s3(1,2) .*t + a_s1(1,1);
qtraj4 = a_s4(1,6) .* t.^5 + a_s4(1,5) .*t.^4 + a_s4(1,4) .*t.^3 + a_s4(1,3) .*t.^2 + a_s4(1,2) .*t + a_s1(1,1);

plot(t,qtraj1)
hold on
plot(t,qtraj2)
hold on
plot(t,qtraj3)
hold on
plot(t,qtraj4)
