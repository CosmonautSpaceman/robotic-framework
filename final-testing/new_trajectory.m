% clear all
% close all
% clc

% t0 = 0; tf = 40; syms q0; v0 = 0; a0 = 0; syms  qf;  vf = 0;  af = 0; 
a_mat_1 = inv([1 t0 t0^2 t0^3 t0^4 t0^5; 
               0 1 2*t0 3*t0^2 4*t0^3 5*t0^4; 
               0 0 2 6*t0 12*t0^2 20*t0^3; 
               1 tf tf^2 tf^3 tf^4 tf^5;
               0 1 2*tf 3*tf^2 4*tf^3 5*tf^4; 
               0 0 2 6*tf 12*tf^2 20*tf^3]); 

syms p0 pf
cart_vec = [p0; v0; a0; pf; vf; af]; pseg = []; pseg_collect = [];
for i = 1:1
    pseg = []
    px = double(subs(a_mat_1 * cart_vec, [p0,pf], [0, 0.6]))
    py = double(subs(a_mat_1 * cart_vec, [p0,pf], [1, 1]))
    pseg = [px';py']
    pseg_collect{i} = pseg
end

pseg = fliplr(pseg)


plen = linspace(0,tf,samples);
time_span = plen;
ts = plen(2) - plen(1);
% step_size = ts;
step_size = ts;

X1 = []

for j =1:length(plen)

%     t_sek=1;
%     sek_nr=ceil(t_sim/t_sek+1e-99);
    t=plen(j)

%     coord_nr=1;
    for i = 1:2
        % i=(sek_nr-1)*2+coord_nr;
        X1(j,i)=pseg(i,1)*t^5+pseg(i,2)*t^4+pseg(i,3)*t^3+pseg(i,4)*t^2+pseg(i,5)*t+pseg(i,6);
    end
end

X1 = [X1, zeros(length(plen),1)]


X1dot = gradient(X1',step_size)'
X1ddot = gradient(X1dot',step_size)'

figure();
subplot(3,1,1),plot(X1);
grid on;
title('C')
subplot(3,1,2),plot(X1dot);
grid on;
title('C dot')
subplot(3,1,3),plot(X1ddot);
grid on;
title('C Ddot')


figure();
plot2(X1);
grid on;
title('Workspace overview')