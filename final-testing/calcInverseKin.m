function qc = calcInverseKin(u)

tx = u(1);
ty = u(2);
tz = u(3);



q1 =  2*atan((7*ty + (- 25*tx^4 - 50*tx^2*ty^2 + 49*tx^2 - 25*ty^4 + 49*ty^2)^(1/2))/(5*tx^2 + 7*tx + 5*ty^2));
q2 =  -2*atan((- 25*tx^2 - 25*ty^2 + 49)^(1/2)/(5*(tx^2 + ty^2)^(1/2)));

if isnan(q1)
    q1 = 0;
end
if isnan(q2)
    q2 = 0;
end


qc = [q1;q2];

end