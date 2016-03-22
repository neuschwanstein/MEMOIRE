from helper.state import loader
from helper.stats import five_stats
from helper.plotting import plt

data,param = loader('Student-Nvarying-Saturated')

# plt.plot(*five_stats(data['CEs_ins'],param['ns']))
# plt.xlabel('$n$')
# plt.ylabel('$\\textrm{Returns (\%)}$')
# plt.axis(ymin=7,ymax=25,xmin=25)
# plt.title('$\\textrm{Five-point summary of in-sample CE distribution -- Saturated features}$')
# plt.show()
# # plt.savefig('./fig/Student-Nvarying-Saturated-INS.pdf',format='pdf')

# plt.plot(*five_stats(data['CEs_oos'],param['ns']))
# plt.xlabel('$n$')
# plt.ylabel('$\\textrm{Returns (\%)}$')
# plt.axis(xmin=25)
# plt.title('$\\textrm{Five-point summary of out-of-sample CE distribution -- Saturated features}$')
# # plt.show()
# plt.savefig('./fig/Student-Nvarying-Saturated-OOS.pdf',format='pdf')

# plt.plot(*five_stats(data['CEs_oos']-data['CEs_ins'],param['ns']))
# plt.xlabel('$n$')
# plt.ylabel('$\\textrm{Returns (\%)}$')
# plt.axis(xmin=25)
# plt.title('$\\textrm{Five-point summary of out-of-sample error CE distribution -- Saturated features}$')
# # plt.show()
# plt.savefig('./fig/Student-Nvarying-Saturated-Error.pdf',format='pdf')

plt.plot(*five_stats(data['CE_star_q_star']-data['CEs_oos'],param['ns']))
plt.xlabel('$n$')
plt.ylabel('$\\textrm{Returns (\%)}$')
plt.axis(xmin=25,ymax=45)
plt.title('$\\textrm{Five-point summary of suboptimality $CE^\\star(q^\\star)-CE^\\star(\\hat q)$ distribution -- Saturated features}$')
# plt.show()
plt.savefig('./fig/Student-Nvarying-Saturated-Suboptimality.pdf',format='pdf')
