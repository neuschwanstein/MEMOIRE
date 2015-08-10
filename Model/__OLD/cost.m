function c = cost(U,p,r)
    global Rf;
    c = pos(U(r.*(r>Rf) + Rf*(r<=Rf)) - U(p.*r + (1-p).*Rf));
end

function result = pos(x)
    result = max(x,0);
end