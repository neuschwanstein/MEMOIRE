import matplotlib.pyplot as plt

class Function(object):
    def _derive(self,x):
        raise NotImplementedError

    def plot(self,x):
        f = self.__call__
        plt.plot(x,f(x))
        plt.show()

def D(f):
    return f._derive
