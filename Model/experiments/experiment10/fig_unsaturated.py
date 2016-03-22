from attrdict import AttrDict

from helper.state import loader
from helper.stats import five_stats
from helper.plotting import plt

data,param = loader('Student-Nvarying-Unsaturated')
data = AttrDict(data)
param = AttrDict(param)

# plt.plot(*five_stats(data.CEs_ins,param.ns))
# plt.xlabel('$n$')
# plt.ylabel('$\\textrm{Returns (\%)}$')
# plt.axis(xmin=25)
# plt.title('$\\textrm{Five-point summary of in-sample CE distribution -- Saturated features}$')
# # plt.show()
# plt.savefig('./fig/Unsaturated/Student-Nvarying-Unsaturated-INS.pdf',format='pdf')

# plt.plot(*five_stats(data.CEs_oos,param.ns))
# plt.xlabel('$n$')
# plt.ylabel('$\\textrm{Returns (\%)}$')
# plt.axis(xmin=25)
# plt.title('$\\textrm{Five-point summary of out-of-sample CE distribution -- Unsaturated features}$')
# # plt.show()
# plt.savefig('./fig/Unsaturated/Student-Nvarying-Unsaturated-OOS.pdf',format='pdf')

# plt.plot(*five_stats(data.CEs_oos-data.CEs_ins,param.ns))
# plt.xlabel('$n$')
# plt.ylabel('$\\textrm{Returns (\%)}$')
# plt.axis(xmin=25,ymin=-20)
# plt.title('$\\textrm{Five-point summary of out-of-sample error $CE^\\star(\\hat q) - \\hat{CE}(\\hat q)$ distribution -- Unsaturated features}$')
# # plt.show()
# plt.savefig('./fig/Unsaturated/Student-Nvarying-Unsaturated-Error.pdf',format='pdf')

plt.plot(*five_stats(data.CE_star_q_star-data.CEs_oos,param.ns))
plt.xlabel('$n$')
plt.ylabel('$\\textrm{Returns (\%)}$')
plt.axis(xmin=25)
plt.title('$\\textrm{Five-point summary of suboptimality $CE^\\star(q^\\star)-CE^\\star(\\hat q)$ distribution -- Unsaturated features}$')
# plt.show()
plt.savefig('./fig/Unsaturated/Student-Nvarying-Unsaturated-Suboptimality.pdf',format='pdf')
