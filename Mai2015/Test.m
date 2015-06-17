function Test
    global Rf p n;
    Rf = 2; p = 10; n = 1000;
    
    n = 100;
    p = 10;
    
    fprintf('This is a test. Now go away.\n');
    X = information();
    R = returns();
    
    solveAlgo(X,R,0)
end

function X = information()
    global p n;
    mu = zeros(1,p);
    Sigma = eye(p);
    X = mvnrnd(mu,Sigma,n);
end

function R = returns()
    global n;
    R = normrnd(5,10,[n,1]);
end

function result = pos(x)
%     result = x.*(x>=0);
    result = max(x,0);
end

function U = linearUtility(r,beta)
    U = r + min(0,beta*r);
end

function U = expUtility(r,mu)
    U = -exp(-mu*r);
end

function c = cost(U,p,r)
    global Rf;
    c = pos(U(r.*(r>Rf) + Rf*(r<=Rf)) - U(p.*r + (1-p).*Rf));
end

function q = solveAlgo(X,R,lambda)
    global p;
    U = @(r) linearUtility(0.5,r);
    
    cvx_begin
        variable q(p)
        minimize(sum(cost(U,X*q,R)) + lambda*norm(q,2))
    cvx_end
end






