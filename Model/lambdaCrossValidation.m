% Lambda cross validation
p = 10;
%t = 5*(rand(p,1)-0.5);
t = [-2.1996 -0.1053 2.3214 2.2308 -0.3148 -0.7829 1.6145 1.6752 2.3670 -2.1281]';
U = @(x) linearUtility(x,0.5);

[Xin,rin] = getDataset(1000,10,t);
[Xout,rout] = getDataset(1000,10,t);

lambdas = 0:0.1:12;
%lambdas = [0];
cin = zeros(size(lambdas));
cout = zeros(size(lambdas));

i=1;
for lambda = lambdas
    q = getq(Xin,rin,lambda);
    cin(i) = sum(cost(U,Xin*q,rin));
    cout(i) = sum(cost(U,Xout*q,rout));
    
    i=i+1;
end

plot(lambdas,cin,lambdas,cout);
legend('Training','Test');