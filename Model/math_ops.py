import numpy as np

class Function(object):
    def _derive(self,x):
        raise NotImplementedError

def D(f):
    return f._derive


class Distribution(object):
    def inverse(self,p):
        raise NotImplementedError

    def _expected_value(self):
        raise NotImplementedError

    def _variance(self):
        raise NotImplementedError

    def _inverse_check(self,p):
        if np.min([p])<0 or np.max([p])>1:
            raise ValueError('p must lie in (0,1)')

    def sample(self,n):
        unif_sample = np.random.uniform(0,1,n)
        return self.inverse(unif_sample)

def E(d):
    try:
        return d.E()
    except AttributeError as e:
        print('Object {} is not a distribution.'.format(d))
        raise e

def Var(d):
    try:
        return d.var()
    except AttributeError as e:
        print('Object {} is not a distribution.'.format(d))
