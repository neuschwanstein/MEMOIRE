function result = getCumulativeReturns(returns)

result = [1 1+returns'];
for i=2:length(result)
    result(i) = result(i-1)*(result(i));
end

end