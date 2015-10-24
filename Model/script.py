import config as cfg
import create_data
import objective as obj
import numpy as np

reload(create_data)

n,p = 500,10

t = create_data.create_rule(p)
create_data.create_data(t,n,p)

X = np.load("Data/dataset.npy")
r = np.load("Data/returns.npy")

lambda0 = 0.01
nLambda = 10
endLambda = 0.01

n = 10000
for l in np.linspace(lambda0,endLambda,nLambda):
    results = np.empty(N)   
    for i in xrange(N):
        q,_ = obj.solve_objective(X,r,l)
        results[i] = q
