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

def plot_histogram_diffMDsize(rule_name,threshold):
    data_path = os.path.join(data_root, 'mdpfc_control_sparsity_4.14_diffMDsize_version2/')




    hp['activation'] ='softplus'
    hp['activation_md'] = 'sig'

    hp['rule_name'] = rule_name#'dm_ctx'


    md_size=[8,32,64,100]

    # scale_list = []

    # succ_0_list = []
    # succ_1_list = []
    # succ_2_list = []
    # succ_3_list = []
    #
    #
    #
    # for model_idx in range(0,70):
    #
    #     succ0 = plot_speed(model_idx=model_idx, md_size=md_size[0])
    #     succ_0_list.append(succ0)
    #     print('model_idx', model_idx,succ0)
    #
    #     # sys.exit(0)
    #
    # for model_idx in range(0, 70):
    #     succ1 = plot_speed(model_idx=model_idx, md_size=md_size[1])
    #     succ_1_list.append(succ1)
    #
    #     # print('model_idx', model_idx, succ1)
    #
    #
    # for model_idx in range(0, 70):
    #     succ2 = plot_speed(model_idx=model_idx, md_size=md_size[2])
    #     succ_2_list.append(succ2)
    #
    # for model_idx in range(0, 70):
    #     succ3 = plot_speed(model_idx=model_idx, md_size=md_size[3])
    #     succ_3_list.append(succ3)
    #     print('model_idx', model_idx, succ3)
    #
    #
    #
    # np.save(data_path + 'succ_0_list.npy', succ_0_list)
    # np.save(data_path + 'succ_1_list.npy', succ_1_list)
    # np.save(data_path + 'succ_2_list.npy', succ_2_list)
    # np.save(data_path + 'succ_3_list.npy', succ_3_list)




    succ_0_list = list(np.load(data_path + 'succ_0_list.npy'))
    succ_1_list = list(np.load(data_path + 'succ_1_list.npy'))
    succ_2_list = list(np.load(data_path + 'succ_2_list.npy'))
    succ_3_list = list(np.load(data_path + 'succ_3_list.npy'))





    succ_0_list_select =  np.array([item for item in succ_0_list if item < threshold])  # Only items greater than 0
    succ_1_list_select =  np.array([item for item in succ_1_list if item < threshold])  # Only items greater than 0
    succ_2_list_select =  np.array([item for item in succ_2_list if item < threshold])  # Only items greater than 0
    succ_3_list_select =  np.array([item for item in succ_3_list if item < threshold])  # Only items greater than 0

    succ_0_list_select = succ_0_list_select[0:50]
    succ_1_list_select = succ_1_list_select[0:50]
    succ_2_list_select = succ_2_list_select[0:50]
    succ_3_list_select = succ_3_list_select[0:50]

    print('=====succ_0_list', succ_0_list_select.shape, succ_0_list_select)
    print('=====succ_1_list', succ_1_list_select.shape, succ_1_list_select)
    print('=====succ_2_list', succ_2_list_select.shape, succ_2_list_select)
    print('=====succ_3_list', succ_3_list_select.shape, succ_3_list_select)

    IT_0 = np.mean(succ_0_list_select)
    IT_1 = np.mean(succ_1_list_select)
    IT_2 = np.mean(succ_2_list_select)
    IT_3 = np.mean(succ_3_list_select)



    IT_std_0 = np.std(succ_0_list_select) / np.sqrt(succ_0_list_select.shape[0])
    IT_std_1 = np.std(succ_1_list_select) / np.sqrt(succ_1_list_select.shape[0])
    IT_std_2 = np.std(succ_2_list_select) / np.sqrt(succ_2_list_select.shape[0])
    IT_std_3 = np.std(succ_3_list_select) / np.sqrt(succ_3_list_select.shape[0])


    IT_mean = [IT_0, IT_1, IT_2, IT_3]
    IT_std = [IT_std_0, IT_std_1, IT_std_2, IT_std_3]

    from scipy import stats
    p01 = stats.ranksums(succ_1_list_select, succ_2_list_select)


    fig = plt.figure(figsize=(3.5, 2))
    ax = fig.add_axes([0.2, 0.1, 0.65, 0.7])

    br1 = 1 * np.arange(len(IT_mean))
    colors = sns.light_palette("seagreen",  n_colors=10)

    #colors = sns.color_palette()  # sns.color_palette("Set2")

    # ax.bar(name_context,IT_mean, yerr =IT_std,color =['black','tab:blue','r'], width=0.3)

    plt.bar(str(md_size[0]), IT_0, yerr=IT_std_0,  color=colors[6], width=0.3, label=str(md_size[0]))
    plt.bar(str(md_size[1]), IT_1, yerr=IT_std_1, color=colors[5],  width=0.3, label=str(md_size[1]))
    plt.bar(str(md_size[2]), IT_2, yerr=IT_std_2, color=colors[4],  width=0.3, label=str(md_size[2]))
    plt.bar(str(md_size[3]), IT_3, yerr=IT_std_3, color=colors[3],  width=0.3, label=str(md_size[3]))


    plt.ylabel('trials', fontsize=12)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title(hp['rule_name'], fontsize=12)
    # plt.ylim([-0.3, 1600])
    # plt.yticks([0, 2, 4, 6], fontsize=13)

    #plt.legend(fontsize=8)
    # fig.savefig(figure_path + hp['rule_name']+'.pdf')
    plt.show()


threshold=4000
plot_histogram_diffMDsize(rule_name='dm_RT',       threshold=threshold)
plot_histogram_diffMDsize(rule_name='dm_fixed',    threshold=threshold)
plot_histogram_diffMDsize(rule_name='dm_ctx',      threshold=threshold)
plot_histogram_diffMDsize(rule_name='multisensory',threshold=threshold)
plot_histogram_diffMDsize(rule_name='GoNogo',      threshold=threshold)



def plot_violin_paper_all(activation,activation_md):
    data_path = os.path.join(data_root, 'mdpfc_control_sparsity_4.14/')


    hp['activation'] ='softplus'
    hp['activation_md'] = 'sig'
    hp['rule_name'] = 'dm_RT'  # 'dm_fixed'#'dm_RT'  # 'dm_fixed'#

    sw_list=[0.4,0.6,0.8]
    #
    # # scale_list = []
    # succ_nofreeze_list = []
    # succ_0_list = []
    # succ_1_list = []
    # succ_2_list = []
    #
    # succ_3_list = []
    # succ_4_list = []
    # succ_5_list = []
    #
    #
    # for model_idx in range(0,46):
    #     print('model_idx',model_idx)
    #     succ0 = plot_speed_rank(model_idx=model_idx, type='type0', freeze='nofreeze',scale=1,sparsity_weight=0.0)
    #     succ_0_list.append(succ0)
    #
    #     # sys.exit(0)
    # for model_idx in range(1, 48):
    #     succ1 = plot_speed_rank(model_idx=model_idx, type='type1', freeze='nofreeze',scale=1,sparsity_weight=0.0)
    #     succ_1_list.append(succ1)
    #
    #
    # for model_idx in range(0, 50):
    #     succ2 = plot_speed(model_idx=model_idx, type='type1', freeze='freeze',scale=30,sparsity_weight=0.2)
    #     succ_2_list.append(succ2)
    #
    #
    # for model_idx in range(0, 50):
    #     succ3 = plot_speed(model_idx=model_idx, type='type1', freeze='freeze',scale=30,sparsity_weight=sw_list[0])
    #     succ_3_list.append(succ3)
    #
    # for model_idx in range(0,50):
    #     print('model_idx',model_idx)
    #     succ4 = plot_speed(model_idx=model_idx, type='type1', freeze='freeze',scale=30,sparsity_weight=sw_list[1])
    #     succ_4_list.append(succ4)
    #
    #     # sys.exit(0)
    # for model_idx in range(0, 50):
    #     succ5 = plot_speed(model_idx=model_idx, type='type1', freeze='freeze',scale=30,sparsity_weight=sw_list[2])
    #     succ_5_list.append(succ5)
    # np.save(data_path + 'succ_0_list.npy', succ_0_list)
    # np.save(data_path + 'succ_1_list.npy', succ_1_list)
    # np.save(data_path + 'succ_2_list.npy', succ_2_list)
    # np.save(data_path + 'succ_3_list.npy', succ_3_list)
    # np.save(data_path + 'succ_4_list.npy', succ_4_list)
    # np.save(data_path + 'succ_5_list.npy', succ_5_list)





    succ_0_list = np.load(data_path + 'succ_0_list.npy')
    succ_1_list = np.load(data_path + 'succ_1_list.npy')
    succ_2_list = np.load(data_path + 'succ_2_list.npy')
    succ_3_list = np.load(data_path + 'succ_3_list.npy')
    succ_4_list = np.load(data_path + 'succ_4_list.npy')
    succ_5_list = np.load(data_path + 'succ_5_list.npy')



    succ_0_list_select =  [item for item in succ_0_list if item < 2000] # Only items greater than 0
    succ_1_list_select =  [item for item in succ_1_list if item < 2000] # Only items greater than 0
    succ_2_list_select =  [item for item in succ_2_list if item < 2000] # Only items greater than 0
    succ_3_list_select =  [item for item in succ_3_list if item < 2000] # Only items greater than 0
    succ_4_list_select =  [item for item in succ_4_list if item < 2000] # Only items greater than 0
    succ_5_list_select =  [item for item in succ_5_list if item < 2000] # Only items greater than 0











    succ_0_list_select = succ_0_list_select[0:40]
    succ_1_list_select = succ_1_list_select[0:40]
    succ_2_list_select = succ_2_list_select[0:40]
    succ_3_list_select = succ_3_list_select[0:40]
    succ_4_list_select = succ_4_list_select[0:40]
    succ_5_list_select = succ_5_list_select[0:40]

    data_violin = np.array([succ_0_list_select, succ_1_list_select,
                            succ_2_list_select,succ_3_list_select,
                            succ_4_list_select,succ_5_list_select]).T

    print('=====succ_0_list', len(succ_0_list), succ_0_list)
    print('=====succ_1_list', succ_1_list)
    print('=====succ_2_list', succ_2_list)
    print('=====succ_3_list', succ_3_list)



    from scipy import stats
    p01 = stats.ranksums(succ_1_list_select, succ_2_list_select)


    fig = plt.figure(figsize=(5, 2.3))
    ax = fig.add_axes([0.17, 0.1, 0.8, 0.7])


    colors = sns.color_palette("muted")  # sns.color_palette("Set2")
    sns.violinplot(data=data_violin, palette=[colors[0], colors[1], colors[2],colors[2],
                                              colors[2], colors[2]],
                   linecolor="k", linewidth=0.5, width=0.4, inner='point', alpha=0.6)

    plt.ylabel('trials', fontsize=12)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title(hp['rule_name']+';  RNN='+hp['activation']+'; FNN='+hp['activation_md']+'\n'+'p='+str(p01[1])+'\n', fontsize=8)
    # plt.ylim([-0.3, 1600])
    # plt.yticks([0, 2, 4, 6], fontsize=13)

    # plt.legend(fontsize=8)
    # fig.savefig(figure_path + hp['rule_name']+'plot_violin_paper_all.png')
    # fig.savefig(figure_path + hp['rule_name']+'plot_violin_paper_all.pdf')
    plt.show()

plot_violin_paper_all(activation='softplus',activation_md='sig')














