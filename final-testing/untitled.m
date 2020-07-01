clear all
close all
clc

x = linspace(0,0.35,100);
xout = linspace(0,0.348,100);

xd = gradient(x)
xdout = gradient(xout)

xdd = gradient(xd)
xddout = gradient(xdout)

Kp = 0.1
Kd = 0.1

I = 0.3
Jinv = 0.1

e = x - xout;
ed = xd - xdout;
xddin = xdd + Kd*ed + Kp*e

Jinv*(xddin - xddout)

