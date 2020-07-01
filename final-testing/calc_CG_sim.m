% function M = calc_inertia(m1, m2, a1, a2, c1, c2, Ixx1, Iyy1, Izz1, Ixx2, Iyy2, Izz2, q1, q2)
% M = [ Izz1 + Izz2 + a1^2*m1 + a1^2*m2 + a2^2*m2 + c1^2*m1 + c2^2*m2 + 2*a1*c1*m1 + 2*a2*c2*m2 + 2*a1*a2*m2*cos(q2) + 2*a1*c2*m2*cos(q2), Izz2 + m2*(a2 + c2)*(a2 + c2 + a1*cos(q2));
%       Izz2 + m2*(a2 + c2)^2 + a1*m2*cos(q2)*(a2 + c2),                      Izz2 + m2*(a2 + c2)^2];
% end


function CG = calc_CG_sim(qc, qcdot)
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
% rb.I =  repmat(eye(3),1,1,2);
% rb.I(:,:,3) = zeros(3,3);

rb.m = [5 5 0];
% In standard DH, COM is mesured respect to its end-effector (using local
% frame). When COM is [0 0 0]', it means the COM is located exactly at the
% end-effector. Therefore, COM usually has negative values, which means the
% COM is behind the end-effector
rb.r = [-0.35 0 0; -0.35 0 0; 0 0 0]';

% -------------------------------------------------------------------------
% Arbitrary trajectory as the inputs: joint position, velocity, and 
% acceleration
% ts = 0.001;
% time_span = 0:ts:1;
% % qc = [pi/3*sin(2*pi*1*time_span)' pi/3*sin(2*pi*1*time_span)' 0*time_span'];
% % qcdot = gradient(qc', ts)';
% % qcddot = gradient(qcdot', ts)';


% qc = [qc(1); qc(2); 0]';
% qcdot = [qcdot(1); qcdot(2); 0]';
qc = [qc(1), qc(2) 0];
qcdot = [qcdot(1), qcdot(2) 0];
qcddot = [0 0 0];
% qcddot = zeros(length(time_span),2);

% -------------------------------------------------------------------------
% Here we go!
CG = invdyn(dh, rb, qc, qcdot, qcddot, [0; 0; 0]);
CG = CG(1:2,:);

% ------------------------------------------------------------------------

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
% i = 1;
% T = calc_transformation(i-1, i, dh, q);
% R = T(1:3, 1:3);
R = eye(3);

w =  R'*zeros(3,1);
wdot = R'*zeros(3,1);
vdot = R'*grav;
Fm = [];
Nm = [];

for k = 1 : size(qc,1)
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
        wdot = R'*(wdot +  z0.*qddot(i) + ...
                    cross(w, z0.*qdot(i)));

        w =  R'*(w + z0.*qdot(i));
        
        vdot = R'*vdot + cross(wdot, p) + ...
            cross(w, cross(w,p));
        
        vcdot = vdot + cross(wdot,rb.r(:,i)) + ...
            cross(w,cross(w,rb.r(:,i)));

        F = rb.m(i)*vcdot;
        N = rb.I(:,:,i)*wdot+cross(w,rb.I(:,:,i)*w);
        
        Fm = [Fm F];
        Nm = [Nm N];
    end

    % Dynamic simulation
    % Backward recursion
    
    for i = N_DOFS:-1:1
        p = [dh.a(i); dh.d(i)*sin(dh.alpha(i)); dh.d(i)*cos(dh.alpha(i))];

        if i < N_DOFS
            T = calc_transformation(i, i+1, dh, q);
            R = T(1:3, 1:3);
            n(:,i) = R*(n(:,i+1) + cross(R'*p, f(:,i+1))) + ...
                cross(rb.r(:,i)+p,Fm(:,i)) + Nm(:,i);
            f(:,i) = R*f(:,i+1) + Fm(:,i);
        else
            n(:,i) = cross(rb.r(:,i)+p,Fm(:,i)) + Nm(:,i);
            f(:,i) = Fm(:,i);
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