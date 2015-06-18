function U = linearUtility(r,beta)
    U = r + min(0,beta*r);
end