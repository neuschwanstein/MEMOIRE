function [X,r,t] = getDataset(n,p,t)
    
    if(~exist('t','var'))
        % p uniformly distributed numbers between -5 and 5.
        t = 5*(rand(p,1)-0.5);
    end
    %Uncorrelated information with mean 0 (variance 1).
    mu = zeros(1,p);
    Sigma = eye(p);
    X = mvnrnd(mu,Sigma,n);
    r = X*t;
    
    % To which we add normal random noise
    r = r + normrnd(0,1,n,1);
end