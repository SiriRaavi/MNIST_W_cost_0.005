import h5py
import numpy as np
import matplotlib.pyplot as plt


# %% loading the training data
h5f = h5py.File('./saved_models/Results/test_Results'
                '.h5', 'r')
all_acc = np.array(h5f['all_acc_episode'])
all_mis = np.array(h5f['mis_episode'])
mislabel_count = np.array(h5f['mislabel_count'])
loss = np.array(h5f['loss_episode'])
h5f.close()

zeros = np.array([mislabel_count[i] for i in range(mislabel_count.shape[0]) if not mislabel_count[i, 0]])
print(zeros)
mis_zeros = np.array([zeros[i] for i in range(zeros.shape[0]) if zeros[i, 1] <= 999])
cor_zeros = np.array([zeros[i] for i in range(zeros.shape[0]) if zeros[i, 1] >= 1000])

mis_loss = np.array([loss[i] for i in range(loss.shape[0]) if all_mis[i, 0] >= 1])
cor_loss = np.array([loss[i] for i in range(loss.shape[0]) if all_mis[i, 0] == 0])

mis_acc1 = np.array([all_acc[i, 0] for i in range(all_acc.shape[0]) if all_mis[i, 0] >= 1])
cor_acc1 = np.array([all_acc[i, 0] for i in range(all_acc.shape[0]) if all_mis[i, 0] == 0])

mis_acc2 = np.array([all_acc[i, 1] for i in range(all_acc.shape[0]) if all_mis[i, 0] >= 1])
cor_acc2 = np.array([all_acc[i, 1] for i in range(all_acc.shape[0]) if all_mis[i, 0] == 0])

mis_acc3 = np.array([all_acc[i, 2] for i in range(all_acc.shape[0]) if all_mis[i, 0] >= 1])
cor_acc3 = np.array([all_acc[i, 2] for i in range(all_acc.shape[0]) if all_mis[i, 0] == 0])

mis_acc4 = np.array([all_acc[i, 3] for i in range(all_acc.shape[0]) if all_mis[i, 0] >= 1])
cor_acc4 = np.array([all_acc[i, 3] for i in range(all_acc.shape[0]) if all_mis[i, 0] == 0])

mis_acc5 = np.array([all_acc[i, 4] for i in range(all_acc.shape[0]) if all_mis[i, 0] >= 1])
cor_acc5 = np.array([all_acc[i, 4] for i in range(all_acc.shape[0]) if all_mis[i, 0] == 0])

mis_acc10 = np.array([all_acc[i, 9] for i in range(all_acc.shape[0]) if all_mis[i, 0] >= 1])
cor_acc10 = np.array([all_acc[i, 9] for i in range(all_acc.shape[0]) if all_mis[i, 0] == 0])

ones = np.array([mislabel_count[i] for i in range(mislabel_count.shape[0]) if mislabel_count[i, 0]])
mis_ones = np.array([ones[i] for i in range(ones.shape[0]) if ones[i, 1] <= 999])
cor_ones = np.array([ones[i] for i in range(ones.shape[0]) if ones[i, 1] >= 1000])




fig = plt.figure()
ax1 = plt.subplot(121)
# ax1.hist(zeros[:, 2], bins=range(0, 200, 2), edgecolor="none", color='darkorchid')
ax1.set_ylim([0, 700])
ax1.set_xlabel('number of episodes')
ax1.set_ylabel('count')
ax1.set_title('Zeros')

ax1.hist(cor_zeros[:, 2], bins=range(0, 500, 5), edgecolor="none", color='limegreen')
ax1.hist(mis_zeros[:, 2], bins=range(0, 500, 5), edgecolor="none", color='orangered')
#
# ax1.text(50, 350, 'Loss: {:.1f}+-({:.1f})'.format(np.mean(mis_loss)*100, np.std(mis_loss)*100), color='orangered',
#          fontsize=10, fontweight='bold')
#
# ax1.text(50, 300, 'ACC2: %{:.1f}+-({:.1f})'.format(np.mean(mis_acc2)*100, np.std(mis_acc2)*100), color='orangered',
#          fontsize=10, fontweight='bold')
# ax1.text(50, 250, 'ACC3: %{:.1f}+-({:.1f})'.format(np.mean(mis_acc3)*100, np.std(mis_acc3)*100), color='orangered',
#          fontsize=10, fontweight='bold')
# ax1.text(50, 200, 'ACC4: %{:.1f}+-({:.1f})'.format(np.mean(mis_acc4)*100, np.std(mis_acc4)*100), color='orangered',
#          fontsize=10, fontweight='bold')
#
# ax1.text(50, 550, 'Loss: {:.1f}+-({:.1f})'.format(np.mean(cor_loss)*100, np.std(cor_loss)*100), color='limegreen',
#          fontsize=10, fontweight='bold')
# ax1.text(50, 500, 'ACC2: %{:.1f}+-({:.1f})'.format(np.mean(cor_acc2)*100, np.std(cor_acc2)*100), color='limegreen',
#          fontsize=10, fontweight='bold')
# ax1.text(50, 450, 'ACC3: %{:.1f}+-({:.1f})'.format(np.mean(cor_acc3)*100, np.std(cor_acc3)*100), color='limegreen',
#          fontsize=10, fontweight='bold')
# ax1.text(50, 400, 'ACC4: %{:.1f}+-({:.1f})'.format(np.mean(cor_acc4)*100, np.std(cor_acc4)*100), color='limegreen',
#          fontsize=10, fontweight='bold')
#
ax2 = plt.subplot(122)
# ax2.hist(ones[:, 2], bins=range(0, 200, 2), edgecolor="none", color='indigo')
ax2.set_ylim([0, 700])
ax2.set_xlabel('number of episodes')
ax2.set_title('Ones')
ax2.set_yticks([])

ax2.hist(cor_ones[:, 2], bins=range(0, 500, 5), edgecolor="none", color='limegreen', label='clean')
ax2.hist(mis_ones[:, 2], bins=range(0, 500, 5), edgecolor="none", color='orangered', label='noisy')
ax2.legend()
plt.show()
#
width = 3.5
height = width / 2
fig.set_size_inches(width, height)
fig.subplots_adjust(left=0.15, bottom=0.25, right=0.95, top=0.95, wspace=0.1, hspace=None)
plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=8)
# fig.savefig('fig4.png')
# fig.savefig('fig5.svg')

print()



