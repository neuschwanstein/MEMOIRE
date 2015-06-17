function q = Test()
    global Rf p n;
    Rf = 2; p = 10; n = 1000;
    
    n = 100;
    p = 10;
    
    [X r] = getDataset(n,p);
    
    q=solveAlgo(X,r,0);
end

function result = pos(x)
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
    
    cvx_begin quiet
        variable q(p)
        minimize(sum(cost(U,X*q,R)) + lambda*norm(q,2))
    cvx_end
end






