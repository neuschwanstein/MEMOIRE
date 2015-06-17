function [X r] = getDataset(n,p)
    
    %Uncorrelated information with mean 0 (variance 1).
    mu = zeros(1,p);
    Sigma = eye(p);
    X = mvnrnd(mu,Sigma,n);
    
    % We suppose r is a linear transformation of X.
    t = [9 -1 -5 -6 10 4 4 3 6 2]'; %randi([-10,10],10,1);
    r = X*t;
    
    % To which we add normal random noise
    r = r + normrnd(0,1,n,1);
end