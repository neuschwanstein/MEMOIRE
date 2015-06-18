function q = getq(X,r,lambda)
    global Rf;
    Rf = 0;
    
    [n,p] = size(X);
    U = @(x) linearUtility(x,0.5);
    
    cvx_begin quiet
        variable q(p)
        minimize(sum(cost(U,X*q,r)) + lambda*norm(q,2))
    cvx_end
end
