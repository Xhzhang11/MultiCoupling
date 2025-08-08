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
data_path = os.path.join(data_root, 'multi_task_mdpfc_3.13/')






def plot_violin(rule_name):
    data_path = os.path.join(data_root, 'multi_task_mdpfc_11.11/')

    # # scale_list = []
    # epoch_0_list = []
    # epoch_1_list = []
    #
    # for model_idx in range(50):
    #     epoch0 = plot_speed(model_idx=model_idx, type='type0', rule_name=rule_name)
    #     epoch_0_list.append(epoch0)
    #
    # for model_idx in range(50):
    #     epoch1 = plot_speed(model_idx=model_idx, type='type1', rule_name=rule_name)
    #     epoch_1_list.append(epoch1)
    #
    # np.save(data_path + rule_name + '_epoch_0_list.npy', epoch_0_list)
    # np.save(data_path + rule_name + '_epoch_1_list.npy', epoch_1_list)

    epoch_0_list = np.load(data_path + rule_name + '_epoch_0_list.npy')
    epoch_1_list = np.load(data_path + rule_name + '_epoch_1_list.npy')



    data_violin = np.array([epoch_0_list, epoch_1_list]).T

    print( '==epoch_0',len(epoch_0_list),epoch_0_list)
    print( '==epoch_1',len(epoch_1_list),epoch_1_list)







    IT_0 = np.mean(epoch_0_list)
    IT_1 = np.mean(epoch_1_list)

    IT_std_0 = np.std(epoch_0_list) / np.sqrt(len(epoch_0_list))
    IT_std_1 = np.std(epoch_1_list) / np.sqrt(len(epoch_1_list))


    IT_mean = [IT_0, IT_1]
    IT_std = [IT_std_0, IT_std_1]

    from scipy import stats
    p01 = stats.ranksums(epoch_0_list, epoch_1_list)
    print('p01',p01)


    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_axes([0.3, 0.1, 0.55, 0.7])


    colors = sns.color_palette()  # sns.color_palette("Set2")
    sns.violinplot(data=data_violin, palette=[colors[0],colors[1]], linecolor="k", linewidth=0.7,
                   width=0.5,inner='point', alpha=0.9)

    #plt.ylabel('trials', fontsize=12)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.title(rule_name+'\n'+': p='+str(p01[1]), fontsize=8)
    # plt.xlim([-0.4, 1.5])
    # plt.yticks([0, 2, 4, 6], fontsize=13)

    #plt.legend(fontsize=8)
    #fig.savefig(figure_path  +'speed.png')
    # fig.savefig(figure_path +'speed_violin_'+rule_name+'.pdf')
    plt.show()
plot_violin(rule_name='GoNogo')
plot_violin(rule_name='dm_fixed')
plot_violin(rule_name='dm_RT')
plot_violin(rule_name='multisensory')
plot_violin(rule_name='dm_ctx')



def plot_box_dm_ctx(rule_name):
    data_path = os.path.join(data_root, 'multi_task_mdpfc_3.13/')

    # # scale_list = []
    # epoch_0_list_type0 = []
    # epoch_1_list_type0 = []
    # epoch_2_list_type0 = []
    # epoch_3_list_type0 = []
    # epoch_4_list_type0 = []
    # epoch_5_list_type0 = []
    #
    # epoch_0_list_type1 = []
    # epoch_1_list_type1 = []
    # epoch_2_list_type1 = []
    # epoch_3_list_type1 = []
    # epoch_4_list_type1 = []
    # epoch_5_list_type1 = []
    #
    #
    #
    #
    # type='type0'
    # for model_idx in range(50):
    #     epoch0 = plot_speed(model_idx=model_idx, batch_size=2,type=type, rule_name=rule_name)
    #     epoch_0_list_type0.append(epoch0)
    #
    # for model_idx in range(50):
    #     epoch1 = plot_speed(model_idx=model_idx, batch_size=4,type=type, rule_name=rule_name)
    #     epoch_1_list_type0.append(epoch1)
    #
    # for model_idx in range(50):
    #     epoch2 = plot_speed(model_idx=model_idx, batch_size=8,type=type, rule_name=rule_name)
    #     epoch_2_list_type0.append(epoch2)
    #
    # for model_idx in range(50):
    #     epoch3 = plot_speed(model_idx=model_idx, batch_size=16,type=type, rule_name=rule_name)
    #     epoch_3_list_type0.append(epoch3)
    #
    # for model_idx in range(50):
    #     epoch4 = plot_speed(model_idx=model_idx, batch_size=32,type=type, rule_name=rule_name)
    #     epoch_4_list_type0.append(epoch4)
    #
    # for model_idx in range(50):
    #     epoch5 = plot_speed(model_idx=model_idx, batch_size=64,type=type, rule_name=rule_name)
    #     epoch_5_list_type0.append(epoch5)
    #
    # type = 'type1'
    # for model_idx in range(50):
    #     epoch0 = plot_speed(model_idx=model_idx, batch_size=2, type=type, rule_name=rule_name)
    #     epoch_0_list_type1.append(epoch0)
    #
    # for model_idx in range(50):
    #     epoch1 = plot_speed(model_idx=model_idx, batch_size=4, type=type, rule_name=rule_name)
    #     epoch_1_list_type1.append(epoch1)
    #
    # for model_idx in range(50):
    #     epoch2 = plot_speed(model_idx=model_idx, batch_size=8, type=type, rule_name=rule_name)
    #     epoch_2_list_type1.append(epoch2)
    #
    # for model_idx in range(50):
    #     epoch3 = plot_speed(model_idx=model_idx, batch_size=16, type=type, rule_name=rule_name)
    #     epoch_3_list_type1.append(epoch3)
    #
    # for model_idx in range(50):
    #     epoch4 = plot_speed(model_idx=model_idx, batch_size=32, type=type, rule_name=rule_name)
    #     epoch_4_list_type1.append(epoch4)
    #
    # for model_idx in range(30,70):
    #     epoch5 = plot_speed(model_idx=model_idx, batch_size=64, type=type, rule_name=rule_name)
    #     epoch_5_list_type1.append(epoch5)
    #
    #
    #
    # np.save(data_path + 'epoch_0_list_type0.npy', epoch_0_list_type0)
    # np.save(data_path + 'epoch_1_list_type0.npy', epoch_1_list_type0)
    # np.save(data_path + 'epoch_2_list_type0.npy', epoch_2_list_type0)
    # np.save(data_path + 'epoch_3_list_type0.npy', epoch_3_list_type0)
    # np.save(data_path + 'epoch_4_list_type0.npy', epoch_4_list_type0)
    #
    # np.save(data_path + 'epoch_0_list_type1.npy', epoch_0_list_type1)
    # np.save(data_path + 'epoch_1_list_type1.npy', epoch_1_list_type1)
    # np.save(data_path + 'epoch_2_list_type1.npy', epoch_2_list_type1)
    # np.save(data_path + 'epoch_3_list_type1.npy', epoch_3_list_type1)
    # np.save(data_path + 'epoch_4_list_type1.npy', epoch_4_list_type1)



    epoch_0_list_type0 = list(np.load(data_path + 'epoch_0_list_type0.npy'))
    epoch_2_list_type0 = list(np.load(data_path + 'epoch_2_list_type0.npy'))
    epoch_3_list_type0 = list(np.load(data_path + 'epoch_3_list_type0.npy'))
    epoch_4_list_type0 = list(np.load(data_path + 'epoch_4_list_type0.npy'))

    epoch_0_list_type1 = list(np.load(data_path + 'epoch_0_list_type1.npy'))
    epoch_2_list_type1 = list(np.load(data_path + 'epoch_2_list_type1.npy'))
    epoch_3_list_type1 = list(np.load(data_path + 'epoch_3_list_type1.npy'))
    epoch_4_list_type1 = list(np.load(data_path + 'epoch_4_list_type1.npy'))




    p02 = stats.ranksums(epoch_0_list_type0, epoch_2_list_type0)
    p23 = stats.ranksums(epoch_2_list_type0, epoch_3_list_type0)
    p34 = stats.ranksums(epoch_3_list_type0, epoch_4_list_type0)

    print('p02', p02)
    print('p23', p23)
    print('p34', p34)


    # print('==epoch_2', np.mean(epoch_2_list_type1), epoch_2_list_type1)
    # print('==epoch_3', np.mean(epoch_3_list_type1), epoch_3_list_type1)
    # print('==epoch_4', np.mean(epoch_4_list_type1), epoch_4_list_type1)
    # print('==epoch_5', np.mean(epoch_5_list_type1), epoch_5_list_type1)



    group0_0 = epoch_0_list_type0
    group0_1 = epoch_2_list_type0
    group0_2 = epoch_3_list_type0
    group0_3 = epoch_4_list_type0

    group1_0 = epoch_0_list_type1
    group1_1 = epoch_2_list_type1
    group1_2 = epoch_3_list_type1
    group1_3 = epoch_4_list_type1
    p1 = stats.ranksums(group0_0,group1_0)
    p2 = stats.ranksums(group0_1,group1_1)
    p3 = stats.ranksums(group0_2,group1_2)
    p4 = stats.ranksums(group0_3,group1_3)

    print('p1=', p1[1])
    print('p2=', p2[1])
    print('p3=', p3[1])
    print('p4=', p4[1])



    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_axes([0.15, 0.1, 0.75, 0.7])
    colors = sns.color_palette()  # sns.color_palette("Set2")
    labels = ['2', '8', '16', '32']

    # Create DataFrame
    df = pd.DataFrame({
        "Value":   group0_0 + group1_0
                 + group0_1 + group1_1
                 + group0_2 + group1_2
                 + group0_3 + group1_3,

        "Group":   ["2"] * (len(group0_0) + len(group1_0))
                 + ["8"] * (len(group0_1) + len(group1_1))
                 + ["16"] * (len(group0_2) + len(group1_2))
                 + ["32"] * (len(group0_3) + len(group1_3)),





        "Subgroup": ["type0"] * len(group0_0) + ["type1"] * len(group1_0) +
                    ["type0"] * len(group0_1) + ["type1"] * len(group1_1) +
                    ["type0"] * len(group0_2) + ["type1"] * len(group1_2) +
                    ["type0"] * len(group0_3) + ["type1"] * len(group1_3)
    })




    # sns.boxplot(x='site', y='value', hue='label', data=df)
    #
    # sns.stripplot(x='site', y='value', hue='label', data=df,
    #               jitter=True, split=True, linewidth=0.5)
    # plt.legend(loc='upper left')

    sns.boxplot(data=df,x="Group", y="Value", hue="Subgroup",
                linewidth=1,
                fliersize=2,
                flierprops={"marker": "o"},
                notch=True, palette=[colors[0], colors[1]],
                saturation=1,
                width=0.5)



    #plt.ylabel('trials', fontsize=12)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.title(rule_name+'\n'+': p='+str(p01[1]), fontsize=8)
    # plt.xlim([-0.4, 1.5])
    plt.ylim([-1000, 30000])

    #plt.legend(fontsize=8)
    #fig.savefig(figure_path  +'speed.png')
    # fig.savefig(figure_path +'batch_size_speed_box_'+rule_name+'_'+'.pdf')
    plt.show()

plot_box_dm_ctx(rule_name='dm_ctx')










