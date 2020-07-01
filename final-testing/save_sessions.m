
disp('Storing simulation in test folder')

the_folder = k+"_test_details";
a = char(the_folder);
mkdir(a);

the_file = strcat(a,'/test_details.txt');
fileID = fopen(the_file,'w');
fprintf(fileID,'The following details summarize the test:\n');
fprintf(fileID,'Simulation time: %f \n',tf);
fprintf(fileID,'Simulation samples: %f \n',samples);
fprintf(fileID,'Cartesian Velocity: %f \n',vf);
fprintf(fileID,'Cartesian Acceleration: %f \n',af);
fprintf(fileID,'Force Constant Disturbance: %d \n',ForceTest(k));
fprintf(fileID,'Force Impulse Disturbance: %d \n',ForceImpulse(k));
fprintf(fileID,'Spring constant: eye(3).* %d \n', KdI(1,1,k));
fprintf(fileID,'Damping constant: eye(3).* %d \n', BdI(1,1,k));
fclose(fileID);

b1 = '/FIG01.png';
b2 = '/FIG02.png';
b3 = '/FIG03.png';
b4 = '/FIG04.png';
b5 = '/FIG05.png';
b6 = '/FIG06.png';
b7 = '/FIG07.png';

fig1t = char(strcat(a,b1));
fig2t = char(strcat(a,b2));
fig3t = char(strcat(a,b3));
fig4t = char(strcat(a,b4));
fig5t = char(strcat(a,b5));
fig6t = char(strcat(a,b6));
fig7t = char(strcat(a,b7));

saveas(fig1,sprintf(fig1t));
saveas(fig2,sprintf(fig2t));
saveas(fig3,sprintf(fig3t));
saveas(fig4,sprintf(fig4t));
saveas(fig5,sprintf(fig5t));
saveas(fig6,sprintf(fig6t));
saveas(fig7,sprintf(fig7t));
