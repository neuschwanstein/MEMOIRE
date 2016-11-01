from helper.state import loader
from helper.stats import five_stats
from helper.plotting import plt

data,param = loader('ut1.00')

dis = data['CE_star'] - data['CE_ins']
# dis = data['CE_star']
plt.plot(*five_stats(dis,param['ps']))
# plt.plot(*five_stats(data['CE_star'],param['ps']))
plt.xlabel('$p$')
plt.ylabel('$\\textrm{Returns}$')
plt.title('$\\textrm{Out of sample CE error}$')
# plt.show()
plt.savefig('./fig/CE_oos_p.pdf',format='pdf')
