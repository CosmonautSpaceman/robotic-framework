% function M = calc_inertia(m1, m2, a1, a2, c1, c2, Ixx1, Iyy1, Izz1, Ixx2, Iyy2, Izz2, q1, q2)
% M = [ Izz1 + Izz2 + a1^2*m1 + a1^2*m2 + a2^2*m2 + c1^2*m1 + c2^2*m2 + 2*a1*c1*m1 + 2*a2*c2*m2 + 2*a1*a2*m2*cos(q2) + 2*a1*c2*m2*cos(q2), Izz2 + m2*(a2 + c2)*(a2 + c2 + a1*cos(q2));
%       Izz2 + m2*(a2 + c2)^2 + a1*m2*cos(q2)*(a2 + c2),                      Izz2 + m2*(a2 + c2)^2];
% end


function M = calc_M()
% -------------------------------------------------------------------------
% Define the DH parameters
N_DOFS = 2;
dh.theta = [0 0 0];
dh.alpha = [0 0 0];
dh.offset = [0 0 0];
dh.d = [0 0 0];
dh.a = [0.7 0.7 0];
dh.type = ['r' 'r' 'r'];

% -------------------------------------------------------------------------
% Rigid body paramaters: inertia, mass, and cener of mass

% Robot Link variables
% m1 = 5; r1 = 0.1; L1 = 0.15;
m2 = 5; r2 = 0.05; L2 = 0.7;
m3 = 5; r3 = 0.05; L3 = 0.7;
% m4 = 4; r4 = 0.03; L4 = 0.4;

% Inertia Matrix
% I_1 = (1/12)*m1*(3*r1^2 + L1^2);
% I_1yy = (1/2)*m1*r1^2;

I_2 = (1/12)*m2*(3*r2^2 + L2^2);
I_2yy = (1/2)*m2*r2^2;

I_3 = (1/12)*m3*(3*r3^2 + L3^2);
I_3yy = (1/2)*m3*r3^2;
% 
% I_4 = (1/12)*m4*(3*r4^2 + L4^2);
% I_4yy = (1/2)*m4*r4^2;


% Remember the order of rigid bodies
I1 = [I_2 0 0; 0 I_2yy 0; 0 0 I_2];
I2 = [I_3 0 0; 0 I_3yy 0; 0 0 I_3];

rb.I =  I1;
rb.I(:,:,2) =  I2;
rb.I(:,:,3) = zeros(3,3);
rb.m = [5 5 0];
rb.r = [-0.35 0 0; -0.35 0 0; 0 0 0]';
ts = 0.001;
time_span = 0:ts:1;
qc = [pi/3*sin(2*pi*1*time_span)' pi/3*sin(2*pi*1*time_span)' 0*time_span'];
% qcdot = gradient(qc', ts)';
% qcddot = gradient(qcdot', ts)'
qcdot = gradient(qc', ts)'.*0;
qcddot1 = [ones(1,1001); zeros(2,1001)]';
qcddot2 = [zeros(1,1001); ones(1,1001); zeros(1,1001)]';

M1 = invdyn(dh, rb, qc, qcdot, qcddot1, [0; 0; 0]);
M2 = invdyn(dh, rb, qc, qcdot, qcddot2, [0; 0; 0]);
% M = [Mi1 Mi2]
M = [];
for i = 1:length(M1)
    M(:,:,i) = [M1(1:2,i) M2(1:2,i)];
end

end

function  T = calc_transformation(from, to, dh, qc)
% Transformation from one joint to another joint
% 0<=from<N_DOFS
% 0<to<=N_DOFS

T = eye(4);
N_DOFS = length(qc);

% Sanity check
if (from >= N_DOFS) || (from < 0) || (to <= 0) || (to >  N_DOFS)
    return;
end

for i = from+1 : to
    if dh.type(i) == 'r'
        dh.theta(i) = qc(i);
    elseif dh.type(i) == 'p'
        dh.d(i) = qc(i);
    end
    
    ct = cos(dh.theta(i) + dh.offset(i));
    st = sin(dh.theta(i) + dh.offset(i));
    ca = cos(dh.alpha(i));
    sa = sin(dh.alpha(i));
    
    T = T * [ ct    -st*ca   st*sa     dh.a(i)*ct ; ...
              st    ct*ca    -ct*sa    dh.a(i)*st ; ...
              0     sa       ca        dh.d(i)    ; ...
              0     0        0         1          ];
end

end

function Q = invdyn(dh, rb, qc, qcdot, qcddot, grav)
% Inverse dynamic with recursive Newton-Euler

if nargin < 6
    grav = [0;0;0];
end

z0 = [0; 0; 1];

for k = 1 : length(qc)  
    q = qc(k, :);
    qdot = qcdot(k, :);
    qddot = qcddot(k, :);

    N_DOFS = length(q);
    
    % ---------------------------------------------------------------------
    % Forward recursion
    for i = 1 : N_DOFS
        T = calc_transformation(i-1, i, dh, q);
        R = T(1:3, 1:3);
        p = [dh.a(i); dh.d(i)*sin(dh.alpha(i)); dh.d(i)*cos(dh.alpha(i))];
        
        if i > 1
            w(:, i) =  R'*(w(:, i-1) + z0.*qdot(i));
            wdot(:, i) = R'*(wdot(:, i-1) +  z0.*qddot(i) + ...
                cross(w(:, i-1), z0.*qdot(i)));
            vdot(:,i) = R'*vdot(:,i-1) + cross(wdot(:,i), p) + ...
                cross(w(:,i), cross(w(:,i),p));
        else
            w(:, i) =  R'*(z0.*qdot(i));
            wdot(:, i) = R'*(z0.*qddot(i));
            vdot(:,i) = R'*grav + cross(wdot(:,i), p) + ...
                cross(w(:,i), cross(w(:,i),p));
        end
    end
    
    % Dynamic simulation
    % Backward recursion
    for i = N_DOFS:-1:1
        p = [dh.a(i); dh.d(i)*sin(dh.alpha(i)); dh.d(i)*cos(dh.alpha(i))];
        
        vcdot = vdot(:,i) + cross(wdot(:,i),rb.r(:,i)) + ...
            cross(w(:,i),cross(w(:,i),rb.r(:,i)));
        
        F = rb.m(i)*vcdot;
        N = rb.I(:,:,i)*wdot(:,i)+cross(w(:,i),rb.I(:,:,i)*w(:,1));
        
        if i < N_DOFS
            T = calc_transformation(i, i+1, dh, q);
            R = T(1:3, 1:3);
            n(:,i) = R*(n(:,i+1) + cross(R'*p, f(:,i+1))) + ...
                cross(rb.r(:,i)+p,F) + N;
            f(:,i) = R*f(:,i+1) + F;
        else
            n(:,i) = cross(rb.r(:,i)+p,F) + N;
            f(:,i) = F;
        end
        
        T = calc_transformation(i-1, i, dh, q);
        R = T(1:3, 1:3);
        
        if dh.type(i) == 't'
            Q(i,k) = f(:,i)'*R'*z0;
        elseif dh.type(i) == 'r'
            Q(i,k) = n(:,i)'*R'*z0;
        end
    end
end
end