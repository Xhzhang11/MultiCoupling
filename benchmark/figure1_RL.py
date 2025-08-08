import sys,os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt


import seaborn as sns
from scipy import stats
import pandas as pd
############ model
hp = {}
hp['root_path'] = os.path.abspath(os.path.join(os.getcwd(),"./"))

data_root = hp['root_path']+'/Datas/'


def plot_violin(task):
    data_path = os.path.join(data_root, 'pyrl_2.24/')

    # # scale_list = []
    # epoch_0_list = []
    # epoch_1_list = []
    #
    # for model_idx in range(10):
    #     epoch0 = plot_speed(model_idx=model_idx, type='type0', task=task)
    #     epoch_0_list.append(epoch0)
    #
    # for model_idx in range(10):
    #     epoch1 = plot_speed(model_idx=model_idx, type='type1', task=task)
    #     epoch_1_list.append(epoch1)

    # np.save(data_path + 'epoch_0_list.npy', epoch_0_list)
    # np.save(data_path + 'epoch_1_list.npy', epoch_1_list)




    epoch_0_list = list(np.load(data_path + 'epoch_0_list.npy'))
    epoch_1_list = list(np.load(data_path + 'epoch_1_list.npy'))

    data_violin = np.array([epoch_0_list, epoch_1_list]).T

    print('==epoch_0', len(epoch_0_list), epoch_0_list)
    print('==epoch_1', len(epoch_1_list), epoch_1_list)

    IT_0 = np.mean(epoch_0_list)
    IT_1 = np.mean(epoch_1_list)

    IT_std_0 = np.std(epoch_0_list) / np.sqrt(len(epoch_0_list))
    IT_std_1 = np.std(epoch_1_list) / np.sqrt(len(epoch_1_list))

    IT_mean = [IT_0, IT_1]
    IT_std = [IT_std_0, IT_std_1]

    from scipy import stats
    p01 = stats.ranksums(epoch_0_list, epoch_1_list)
    print('p01', p01)

    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_axes([0.3, 0.1, 0.55, 0.7])

    colors = sns.color_palette()  # sns.color_palette("Set2")
    sns.violinplot(data=data_violin, palette=[colors[0], colors[1]], linecolor="k",
                   linewidth=0.7, width=0.5, inner='point', alpha=0.9)

    plt.ylabel('trials', fontsize=12)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title(task + '\n' + ': p=' + str(p01[1]), fontsize=8)
    # plt.xlim([-0.4, 1.5])
    # plt.yticks([0, 2, 4, 6], fontsize=13)

    # plt.legend(fontsize=8)
    # fig.savefig(figure_path  +'speed.png')
    # fig.savefig(figure_path + 'speed_violin_' + task + '.pdf')
    plt.show()


plot_violin(task='go')
plot_violin(task='rdmfixed')
plot_violin(task='rdmrt')
plot_violin(task='multisensory')
plot_violin(task='mante')




