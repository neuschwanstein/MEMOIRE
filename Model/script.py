import config as cfg
import create_data

n,p = 1000,100

t = create_data.create_rule(p)
create_data.create_data(t,n,p)

X = np.load("Data/dataset.npy")
r = np.load("Data/returns.npy")

lambda0 = 0.01
deltaLamba = 0.02

N = 10000

for l in xrange(lambda0, 1, deltaLambda):
    cfg.Lambda = l
    q,_ = solve_objective(X,r)
