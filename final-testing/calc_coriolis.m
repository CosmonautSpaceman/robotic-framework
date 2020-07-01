function C = calc_coriolis(m1, m2, a1, a2, c1, c2, q1, q2, q1d, q2d)

C = [ -a1*m2*q2d*sin(q2)*(a2 + c2), -a1*m2*sin(q2)*(a2 + c2)*(q1d + q2d);
        a1*m2*q1d*sin(q2)*(a2 + c2),                                    0];
end