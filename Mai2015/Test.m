beta = 1;
rc=0;
Rf = 0.05;

lambda = 10;

p=10;
n=1000;
mu=zeros(p,1);
sigma=diag(ones(p,1));
S = mvnrnd(mu,sigma,n);
w = [8/10 1/45*ones(1,9)]';

r = S*w;
r = r.*lognrnd(0,1,n,1); % Noise data 
%r = r.*normrnd(1,5,n,1); %Noise up data


U = @(p) p-rc + min((beta-1)*(p-rc),0);
Uhat = @(q) sum(U(r.*(S*q) + Rf*(1-S*q)));

cvx_begin
    variable q(p)
    maximize(Uhat(q) - 1000*norm(q,2))
%     subject to
%         norm(q,2) <= 1        
cvx_end

q