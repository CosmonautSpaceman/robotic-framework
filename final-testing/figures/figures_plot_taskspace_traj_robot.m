disp('Computing the workspace trajectory:')

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


robot2 = SerialLink( [ Revolute('a', 0.7,'m',5,'r',[-0.35; 0; 0], 'I', I1) ...
    Revolute('a', 0.7,'m',5,'r',[-0.35; 0; 0],  'I', I2)],'name', 'my robot');
robot2.gravity = [0; 0; -9.8];

p1 = linspace(0,pi,100);
p2 = linspace(0,deg2rad(150),100);
p3 = linspace(pi,0,100);
p4 = linspace(deg2rad(150),deg2rad(-150),100);

Tcollect = [];
seg1 = [];
seg2 = [];
seg3 = [];
seg4 = [];
seg5 = [];

for i= 1:length(p1)
    [T,A] = robot2.fkine([p1(i), 0]);
    segrange = A(2).transl;
    seg1 = [segrange' seg1];
end
% Seg 2
for i= 1:length(p1)
    [T,A] = robot2.fkine([p1(end),p2(i)]);
    segrange = A(2).transl;
    seg2 = [segrange' seg2];
end
% Seg 3
for i= 1:length(p1)
    [T,A] = robot2.fkine([p3(i),p2(end)]);
    segrange = A(2).transl;
    seg3 = [segrange' seg3];
end
% Seg 4
for i= 1:length(p1)
    [T,A] = robot2.fkine([p3(end),p4(i)]);
    segrange = A(2).transl;
    seg4 = [segrange' seg4];
end
% Seg 5
for i= 1:length(p1)
    [T,A] = robot2.fkine([p1(i),p4(end)]);
    segrange = A(2).transl;
    seg5 = [segrange' seg5];
end
seg1 = seg1';
seg2 = seg2';
seg3 = seg3';
seg4 = seg4';
seg5 = seg5';
seg_total = [seg1;seg2;seg3;seg4;seg5];

disp('Plotting the robot trajectory without the robot:');
% Plot without the robot

fig3 = figure();
plot2(seg1,'color',[0 0.4470 0.7410]);
hold on;
plot2(seg2,'color',[0 0.4470 0.7410]);
hold on;
plot2(seg3,'color',[0 0.4470 0.7410]);
hold on;
plot2(seg4,'color',[0 0.4470 0.7410]);
hold on;
plot2(seg5,'color',[0 0.4470 0.7410]);
view(2);
hold on;
grid on;
h = title('Reachable Workspace (Max/min)','interpreter','latex');
h.FontSize=14;
hold on;
plot2(X1,'ro');
hold on;
plot2(x_out','b.');
xlabel('x-axis [m]','interpreter','latex');
ylabel('y-axis [m]','interpreter','latex');


disp('Plotting the robot trajectory with robot depiction:');
% Plot with the robot

fig4 = figure();
robot2.plot([0,0],'jvec','arrow','workspace', [-1.5, 1.5 -1 1.5 -1 1],'noname');
hold on;
plot2(seg1,'color',[0 0.4470 0.7410]);
hold on;
plot2(seg2,'color',[0 0.4470 0.7410]);
hold on;
plot2(seg3,'color',[0 0.4470 0.7410]);
hold on;
plot2(seg4,'color',[0 0.4470 0.7410]);
hold on;
plot2(seg5,'color',[0 0.4470 0.7410]);
hold on;
grid on;
h = title('Reachable Workspace (Max/min)','interpreter','latex');
h.FontSize=14;
hold on;
plot2(X1,'ro');
hold on;
plot2(x_out','b.');

xlabel('x-axis [m]','interpreter','latex');
ylabel('y-axis [m]','interpreter','latex');