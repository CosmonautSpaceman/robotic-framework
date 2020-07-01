function C=C_path(t_sim)

global GM 

t_sek=2;
sek_nr=ceil(t_sim/t_sek+1e-99);
t=t_sim-(sek_nr-1)*t_sek;

coord_nr=1;
i=(sek_nr-1)*2+coord_nr;
C(coord_nr)=GM(i,1)*t^5+GM(i,2)*t^4+GM(i,3)*t^3+GM(i,4)*t^2+GM(i,5)*t+GM(i,6);

coord_nr=2;
i=(sek_nr-1)*2+coord_nr;
C(coord_nr)=GM(i,1)*t^5+GM(i,2)*t^4+GM(i,3)*t^3+GM(i,4)*t^2+GM(i,5)*t+GM(i,6);

coord_nr=3;
C(coord_nr)=.2;

coord_nr=4;
C(coord_nr)=0;

coord_nr=5;
C(coord_nr)=0;

coord_nr=6;
C(coord_nr)=-1;
