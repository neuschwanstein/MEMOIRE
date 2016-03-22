import numpy as np
import matplotlib.pyplot as plt

from helper.stats import five_stats
from helper.state import loader

plt.rc('text',usetex=True)

data,param = loader('graph1')

plt.plot(*five_stats(data['CE_ins'],param['λs']))
plt.xlabel('$p$')
plt.ylabel('$\\textrm{Returns}$')
plt.title('$\\textrm{Five-point summary of in-sample CE distribution}$')
plt.show()
# plt.savefig('./fig/CE_ins_λ.pdf',format='pdf')
