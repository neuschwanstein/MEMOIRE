import numpy as np
import matplotlib.pyplot as plt

from helper.stats import five_stats

plt.rc('text',usetex=True)

CE_ins = np.load('CE_ins.npy')
CE_star = np.load('CE_star.npy')
ns = np.arange(25,2025,25)

# Graph 1 - Insample CE
# plt.plot(*five_stats(CE_ins,ns))
# plt.axis(xmin=25,xmax=2000,ymax=25)
# plt.xlabel('$\\textrm{Sample size}$')
# plt.ylabel('$\\textrm{Returns}$')
# plt.title('$\\textrm{Five-point summary of in-sample CE distribution}$')
# # plt.show()
# plt.savefig('./fig/CE_ins.pdf',format='pdf')

# Graph 2 - Absolute distance between CE and insample CE
dis = sbpt
plt.plot(*five_stats(dis,ns))
# plt.axis(xmin=25,ymin=-7,ymax=2)
plt.xlabel('$\\textrm{Sample size}$')
plt.ylabel('$\\textrm{Risks}$')
plt.title('$\\textrm{Five-point summary of }|R(q^\star) - R(\hat q)|$')
plt.show()
# plt.savefig('./fig/CE_star-CE_ins.pdf')
