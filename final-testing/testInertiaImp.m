function y = testInertiaImp(u)

% Robot Link variables
m2 = 5; r2 = 0.05; L2 = 0.7;
m3 = 5; r3 = 0.05; L3 = 0.7;

I_2 = (1/12)*m2*(3*r2^2 + L2^2);
I_2yy = (1/2)*m2*r2^2;

I_3 = (1/12)*m3*(3*r3^2 + L3^2);
I_3yy = (1/2)*m3*r3^2;


% Remember the order of rigid bodies
I1 = [I_2 0 0; 0 I_2yy 0; 0 0 I_2]
I2 = [I_3 0 0; 0 I_3yy 0; 0 0 I_3]

robot2 = SerialLink( [ Revolute('a', 0.7,'m',5,'r',[-0.35; 0; 0], 'I', I1) ...
    Revolute('a', 0.7,'m',5,'r',[-0.35; 0; 0],  'I', I2)],'name', 'my robot')

qc = [u(1), u(2)];
xdd = [u(3), u(4) u(5), u(6) u(7), u(8)]'

M = robot2.inertia(qc)
M = [M zeros(2,1); zeros(1,3)]
% J = robot2.jacob0(qc)

Km1 = eye(3).*[m2;m3;0]


Km = [Km1 zeros(3,3);zeros(3,3) M];

% inv(M)

% Lambda = inv(J*inv(M)*J')
% 
% Lambda = (inv(J))' * J * inv(J);

% LambdaInv = inv(Lambda);
% Lambda = robot2.cinertia(qc)

y = Km*xdd;
end


