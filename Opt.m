xs = features;
rs = returns';

b1 = 0.2;
b2 = 1;

Rf = log(1.02)/252;
rc = log(1.15)/252;

[n p] = size(xs);

U = @(p) b1*(p-rc) + min(0, (b2-b1)*(p-rc));

cvx_begin
    variable q(p)    
    maximize(sum(U(rs'.*(xs*q) + Rf*(1-xs*q))))
cvx_end
