from helper.state import loader
from helper.stats import five_stats
from helper.plotting import plt

data,param = loader('ut1.00')

plt.plot(*five_stats(data['CE_ins'],param['ps']))
plt.xlabel('$p$')
plt.ylabel('$\\textrm{Returns}$')
plt.title('$\\textrm{Five-point summary of in-sample CE distribution}$')
# plt.show()
plt.savefig('./fig/CE_ins_p.pdf',format='pdf')
