from cvxpy import *
import numpy as np

r = 1
x1 = np.array([1,1,1])

q1 = Variable(3)

objective = Maximize(x1*q1*r - 0.8*power(norm(x1*q1,1),2))
problem = Problem(objective)

result = problem.solve()
print "Optimal value: ", problem.value
