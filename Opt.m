if (~exist('features','var'))
    xs = importdata('features.csv');
    rs = importdata('returns.csv');
end

bs = linspace(0,1,5);
%bs = [0 1];
returns = [];

for b=bs
    q = getOpt(0,b,xs,rs);

    pfReturns = rs.*(xs*q) + Rf*(1-xs*q);
    totalReturns = getCumulativeReturns(pfReturns);
    returns = [returns totalReturns'];
end
% 
% pfReturns = rs.*(xs*q) + Rf*(1-xs*q);
% totalReturn = getCumulativeReturns(pfReturns);
% totalRawReturns = getCumulativeReturns(rs);

semilogy([totalRawReturns' returns])
xlim([0 length(totalRawReturns)]);
%legend('Portfolio Return', 'Raw Returns','Location','northwest');