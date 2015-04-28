if (~exist('features','var'))
    xs = importdata('features.csv');
    rs = importdata('returns.csv');
end

totalRawReturns = getCumulativeReturns(rs);

% bs = linspace(0,1,5);
bs = [1];
returns = [];
Rf = log(1.02)/252;

for b=bs
    q = getOpt(Rf,b,xs,rs);

    pfReturns = rs.*(xs*q) + Rf*(1-xs*q);
    totalReturns = getCumulativeReturns(pfReturns);
    returns = [returns totalReturns'];
end

semilogy([totalRawReturns' returns])
xlim([0 length(totalRawReturns)]);
%legend('Portfolio Return', 'Raw Returns','Location','northwest');