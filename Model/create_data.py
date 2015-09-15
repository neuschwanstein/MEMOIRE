import numpy as np
import config as cfg

n,p = cfg.n,cfg.p
    
def create_data(n,p):
    dataset = np.random.randn(n,p)
    #rule = np.random.normal(5,5,p) # High var to have more 'extreme' values
    rule = np.random.normal(*(cfg.t_mean_and_variance + [p]))
    # noise = np.random.normal(0,3,n)
    noise = np.random.normal(*(cfg.noise_mean_and_variance + [n]))

    returns = np.dot(dataset,rule) + noise

    np.save("Data/returns.npy", returns)
    np.save("Data/rule.npy", rule)
    np.save("Data/dataset.npy", dataset)

if (__name__ == "__main__"):
    create_data(n,p)







# import primes

# from compiler.ast import flatten

# def nest_list(f,expr,n):
#     val = f(expr)
#     if n == 1:
#         return val
#     return [val, nest_list(f,val,n-1)]
        

# def integer_list(beg,end,seed):
#     m = end-beg
#     p = primes.next_prime(int(m/4))
#     next_point = lambda n: (n+p) % m + beg
#     # return nest_list(next_point,int(m/2),m)
#     return flatten(nest_list(next_point,int(m/2),seed))
