function q = getOpt(rc,b2,xs,rs)

Rf = log(1.02)/252;
[n p] = size(xs);
b1=1;
U = @(p) b1*(p-rc) + min(0, (b2-b1)*(p-rc));

cvx_begin
    variable q(p)
    maximize(sum(U(rs.*(xs*q) + Rf*(1-xs*q))))
    subject to
        norm(q,2) <= 1
cvx_end

end