import matplotlib.pyplot as plt
import model.distrs as ds

class Function(object):
    def _derive(self,x):
        raise NotImplementedError

    def _create_distribution(self,X):
        return ds.FunctionDistribution(X,self)

    def plot(self,x):
        f = self.__call__
        plt.plot(x,f(x))
        plt.show()

    def __call__(self,x):
        if isinstance(x,ds.Distribution):
            return self._distribution_call(x)
        else:
            return self._call(x)

    def _distribution_call(self,X):
        return ds.FunctionDistribution(X,self)

def D(f):
    return f._derive
