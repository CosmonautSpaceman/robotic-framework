function Q = RNE_sim(u)
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

I_2 = (1/12)*m2*(3*r2^2 + L2^2);
I_2yy = (1/2)*m2*r2^2;

I_3 = (1/12)*m3*(3*r3^2 + L3^2);
I_3yy = (1/2)*m3*r3^2;

% Remember the order of rigid bodies
I1 = [I_2 0 0; 0 I_2yy 0; 0 0 I_2];
I2 = [I_3 0 0; 0 I_3yy 0; 0 0 I_3];


rb.I =  I1;
rb.I(:,:,2) =  I2;
rb.I(:,:,3) = zeros(3,3);

rb.m = [5 5 0];
rb.r = [-0.35 0 0; -0.35 0 0; 0 0 0]';

% ts = 0.001;
% time_span = 0:ts:1;
% qc = [pi/3*sin(2*pi*1*time_span)' pi/3*sin(2*pi*1*time_span)' 0*time_span'];
% qcdot = gradient(qc', ts)';
% qcddot = gradient(qcdot', ts)';

qc = [u(1) u(2) 0];
qcdot = [u(3) u(4) 0];
qcddot = [u(5) u(6) 0];

% qc = [[qc(1) qc(3)]' [qc(2) qc(4)]' zeros(2,1)];
% qcdot = [[qcdot(1) qcdot(3)]' [qcdot(2) qcdot(4)]' zeros(2,1)];
% qcddot = [[qcddot(1) qcddot(3)]' [qcddot(2) qcddot(4)]' zeros(2,1)];

Q = invdyn(dh, rb, qc, qcdot, qcddot, [0; 0; -9.8]);
Q = Q(1:2,:);

end

function  T = calc_transformation(from, to, dh, qc)

T = eye(4);
N_DOFS = length(qc);

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
        
%         a = vdot
%         test = rb.r(:,i)
%         bn = cross(wdot,rb.r(:,i))
%         e1 = cross(w,cross(w,rb.r(:,i)))
        
        
        vcdot = vdot + cross(wdot,rb.r(:,i)) + ...
            cross(w,cross(w,rb.r(:,i)));

        F = rb.m(i)*vcdot;
        
        
        c = rb.I(:,:,i)*wdot;
        p2 = cross(w,rb.I(:,:,i)*w);
        N = c + p2;
        
%         N = rb.I(:,:,i)*wdot+cross(w,rb.I(:,:,i)*w)
        
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
            
%             b2 = cross(R'*p, f(:,i+1))
%             
%             a = R*(n(:,i+1) +  cross(R'*p, f(:,i+1)));
%             c = cross(rb.r(:,i)+p,Fm(:,i));
%             d = Nm(:,i);
%             
            
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