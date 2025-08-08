import sys,os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt


import seaborn as sns
from scipy import stats
import pandas as pd
hp = {}
hp['root_path'] = os.path.abspath(os.path.join(os.getcwd(),"./"))

data_root = hp['root_path']+'/Datas/'



def plot_loss():
    data_path = os.path.join(data_root, 'mdpfc_control_sparsity_4.17/')

    hp['rule_name'] = 'dm_fixed'  # 'dm_fixed'#'dm_RT'  # 'dm_fixed'#

    accu_type0_list = []
    accu_type1_list = []


    # idx=17
    # accu0 = get_loss(model_idx=idx, type='type0')
    # accu1 = get_loss(model_idx=16, type='type1')
    #
    # np.save(data_path + 'accu0.npy', accu0)
    # np.save(data_path + 'accu1.npy', accu1)






    accu0 = list(np.load(data_path + 'accu0.npy'))
    accu1 = list(np.load(data_path + 'accu1.npy'))







    # for i in range(0,70):
    #     fig = plt.figure(figsize=(4, 2))
    #     ax = fig.add_axes([0.3, 0.1, 0.55, 0.7])
    #
    #     plt.plot(accu_type0_list[i][150:], c='tab:blue')
    #     # plt.plot(accu_type1_list[i],c='tab:orange')
    #     fig.savefig(figure_path +'type0_'+ str(i) + '.png')
    #     # plt.show()


    fig = plt.figure(figsize=(3, 2.2))
    ax = fig.add_axes([0.25, 0.2, 0.65, 0.6])
    plt.plot(accu0[0:],c='tab:blue')
    plt.plot(accu1[0:], c='tab:orange')

    plt.scatter(5, accu0[5], s=20,c='blue',zorder=3)
    plt.scatter(150, accu0[150], s=20,c='blue',zorder=3)
    plt.scatter(165, accu0[165], s=20,c='blue',zorder=3)
    plt.scatter(200, accu0[200], s=20, c='blue', zorder=3)

    plt.scatter(5,accu1[5],s=20,c='red',zorder=3)
    plt.scatter(25, accu1[25], s=20,c='red',zorder=3)
    plt.scatter(36, accu1[36], s=20,c='red',zorder=3)
    plt.scatter(45, accu1[45], s=20, c='red', zorder=3)
    # plt.plot(accu_type1_list[i],c='tab:orange')
    # plt.title('loss'+str(idx))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    ticks = [0, 50,100, 150,200]
    plt.xticks(ticks, [int(t*20) for t in ticks])  # Show [0, 5, 10] instead of [0, 50, 100]

    plt.xlabel('Trials')
    plt.ylabel('Loss')



    # fig.savefig(figure_path +'loss'+ '.pdf')
    plt.show()

plot_loss()
