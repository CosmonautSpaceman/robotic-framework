syms a1 a2 g real
syms c1 c2 m1 m2 real

%syms Iyy1 Iyy2 b1 b2 real
Iyy1 = 0
Iyy2 = 0
b1 = 0
b2 = 0

twolink = SerialLink([
    Revolute('d', 0, 'a', a1, 'alpha', 0, 'm', m1, 'r', [c1 0 0], 'I', [0 Iyy1 0], 'B', b1, 'G', 1, 'Jm', 0, 'standard')
    Revolute('d', 0, 'a', a2, 'alpha', 0, 'm', m2, 'r', [c2 0 0], 'I', [0 Iyy2 0], 'B', b2, 'G', 1, 'Jm', 0, 'standard')
    ], ...
    'name', 'two link', ...
    'comment', 'from Spong, Hutchinson, Vidyasagar');
twolink = twolink.sym();
twolink.gravity = [0; 0; g];
twolink.base = trotx(sym('pi')/2);

syms q1 q2 q1d q2d q1dd q2dd real