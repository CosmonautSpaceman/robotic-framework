function qddr = inverseDynamics(Q,M,CGJF)
qddr = inv(M)*(Q -CGJF);
end