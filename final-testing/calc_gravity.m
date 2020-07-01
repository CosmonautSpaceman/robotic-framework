function G = calc_gravity(g, m1, m2, a1, a2, c1, c2, q1, q2)
G = [ g*(a1*m1*cos(q1) + a1*m2*cos(q1) + c1*m1*cos(q1) + a2*m2*cos(q1 + q2) + c2*m2*cos(q1 + q2)) g*m2*cos(q1 + q2)*(a2 + c2)];
end