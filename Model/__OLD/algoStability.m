p = 3;
lambda = 0;
m = 15;

U = @(x) linearUtility(x,0.5);
[XLarge, rLarge, tLarge] = getDataset(1000000,p);

ns = ceil(linspace(50,100,10));
betas = zeros(size(ns));

i = 1;
for n=ns
    disp(['n=',num2str(n)]);
    subBetas = zeros(m,1);
    
    for j=1:m
        disp(['j=',num2str(j)]);
        [X,r,t] = getDataset(n,p);

        q = getq(X,r,0);
        qMinus = getq(X(2:end,:),r(2:end),0);

        subBetas(i) = max(abs(cost(U,XLarge*q,rLarge) - cost(U,XLarge*qMinus,rLarge)));
    end
    
    betas(i) = max(subBetas);
    i = i+1;
end