if (~exist('features','var'))
    xs = importdata('features.csv');
    rs = importdata('returns.csv');
end

% xs = xs(1:500,:);
% rs = rs(1:500,:);

b1 = 1;
b2 = 1;

Rf = log(1.02)/252;
%rc = log(1.50)/252;
rc = 0;
lambda=0.5;

[n p] = size(xs);

U2 = @(p) b1*(p-rc) + min(0, (b2-b1)*(p-rc));

cvx_begin
    variable q(p)
    maximize(sum(U2(rs.*(xs*q) + Rf*(1-xs*q))))
    subject to
        norm(q,2) <= 1
cvx_end

pfReturns = rs.*(xs*q) + Rf*(1-xs*q);
totalReturn = getCumulativeReturns(pfReturns);
totalRawReturns = getCumulativeReturns(rs);

semilogy([totalReturn' totalRawReturns'])
xlim([0 length(totalReturn)]);
legend('Portfolio Return', 'Raw Returns','Location','northwest');