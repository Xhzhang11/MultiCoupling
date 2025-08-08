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
data_path = os.path.join(data_root, 'outcome_tactile_EI_ver2_6.3/')





def switching_sem_combine(task,type):
    hp['type'] = type
    hp['task']=task

    P_choice_list=[]
    P_correct_list = []

    # idx_list = [1,2,4,6,8,  10,13,5,18,19]#5
    # models_0 = idx_list#[1,2,4,5,6,   8,10,13,0,16]
    # models_1 = idx_list#[1,2,4,5,6,   8,10,13,0,16]#[1,2,4,5,6,   8,10,13,0,16]
    #
    # P_choice_list_0 = []
    # P_correct_list_0 = []
    #
    # P_choice_list_1 = []
    # P_correct_list_1 = []
    #
    # for model_idx in np.array(models_0):
    #     pre_switch = 20
    #     post_switch = 35
    #     P_choice_conA_0,P_correct_conA_0,_ = get_log(model_idx=model_idx, context='conA',p_out_uncert=0.8)
    #     P_choice_conB_0,P_correct_conB_0,_ = get_log(model_idx=model_idx, context='conB',p_out_uncert=0.8)
    #
    #     P_choice_0 = P_choice_conA_0[-pre_switch:]+P_choice_conB_0[0:post_switch]
    #     P_correct_0 = P_correct_conA_0[-pre_switch:]+P_correct_conB_0[0:post_switch]
    #
    #
    #     print('P_choice_conA[-15:]',len(P_choice_conA_0[-pre_switch:]))
    #     print('P_choice_conB[-15:]', len(P_choice_conB_0[0:post_switch]))
    #
    #     print('P_choice', len(P_choice_0))
    #
    #     P_choice_list_0.append(P_choice_0)
    #     P_correct_list_0.append(P_correct_0)
    #
    #
    # for model_idx in np.array(models_1):
    #     pre_switch = 20
    #     post_switch = 29
    #     P_choice_conA_1,P_correct_conA_1,_ = get_log(model_idx=model_idx, context='conA',p_out_uncert=0.9)
    #     P_choice_conB_1,P_correct_conB_1,_ = get_log(model_idx=model_idx, context='conB',p_out_uncert=0.9)
    #
    #     P_choice_1 = P_choice_conA_1[-pre_switch:]+P_choice_conB_1[0:post_switch]
    #     P_correct_1 = P_correct_conA_1[-pre_switch:]+P_correct_conB_1[0:post_switch]
    #
    #
    #     print('P_choice_conA[-15:]',len(P_choice_conA_1[-pre_switch:]))
    #     print('P_choice_conB[-15:]', len(P_choice_conB_1[0:post_switch]))
    #
    #     print('P_choice_1', len(P_choice_1))
    #
    #     P_choice_list_1.append(P_choice_1)
    #     P_correct_list_1.append(P_correct_1)
    #
    #
    #
    #
    # # Convert to numpy arrays
    #
    # P_choice_numpy_0 = np.array(P_choice_list_0)
    # P_choice_numpy_1 = np.array(P_choice_list_1)
    # P_correct_numpy_0 = np.array(P_correct_list_0)
    # P_correct_numpy_1 = np.array(P_correct_list_1)
    #
    # np.save(data_path + 'P_choice_numpy_0.npy',P_choice_numpy_0)
    # np.save(data_path + 'P_choice_numpy_1.npy', P_choice_numpy_1)
    # np.save(data_path + 'P_correct_numpy_0.npy', P_correct_numpy_0)
    # np.save(data_path + 'P_correct_numpy_1.npy', P_correct_numpy_1)


    P_choice_numpy_0 = np.load(data_path + 'P_choice_numpy_0.npy')
    P_choice_numpy_1 = np.load(data_path + 'P_choice_numpy_1.npy')

    P_correct_numpy_0 = np.load(data_path + 'P_correct_numpy_0.npy')
    P_correct_numpy_1 = np.load(data_path + 'P_correct_numpy_1.npy')




    # Compute mean and SEM
    P_choice_mean_0 = P_choice_numpy_0.mean(axis=0)
    P_correct_mean_0 = P_correct_numpy_0.mean(axis=0)
    P_choice_sem_0 = P_choice_numpy_0.std(axis=0, ddof=1) / np.sqrt(P_choice_numpy_0.shape[0])
    P_correct_sem_0 = P_correct_numpy_0.std(axis=0, ddof=1) / np.sqrt(P_correct_numpy_0.shape[0])


    # Compute mean and SEM
    P_choice_mean_1 = P_choice_numpy_1.mean(axis=0)
    P_correct_mean_1 = P_correct_numpy_1.mean(axis=0)
    P_choice_sem_1 = P_choice_numpy_1.std(axis=0, ddof=1) / np.sqrt(P_choice_numpy_1.shape[0])
    P_correct_sem_1 = P_correct_numpy_1.std(axis=0, ddof=1) / np.sqrt(P_correct_numpy_1.shape[0])




    # Plot with shaded error
    fig = plt.figure(figsize=(3.5, 2.5))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.65])
    x_0 = np.arange(len(P_correct_mean_0))
    x_1 = np.arange(len(P_correct_mean_1))

    # colors = sns.color_palette()

    colors = sns.color_palette("Set2")#sns.color_palette("hls", 8)

    # # Plot means and SEM shaded area
    ax.plot(x_0, P_correct_mean_0, c=colors[0], label='0.8')
    ax.fill_between(x_0, P_correct_mean_0 - P_correct_sem_0, P_correct_mean_0 + P_correct_sem_0,
                    color=colors[0], alpha=0.3)
    #
    # ax.plot(x_0, P_choice_mean_0, c='gray', label='P_choice')
    # ax.fill_between(x_0, P_choice_mean_0 - P_choice_sem_0, P_choice_mean_0 + P_choice_sem_0,
    #                 color='gray', alpha=0.3)





    # Plot means and SEM shaded area
    ax.plot(x_1, P_correct_mean_1, c=colors[1], label='P_correct')
    ax.fill_between(x_1, P_correct_mean_1 - P_correct_sem_1, P_correct_mean_1 + P_correct_sem_1,
                    color=colors[1], alpha=0.3)

    # ax.plot(x_1, P_choice_mean_1, c='gray', label='P_correct')
    # ax.fill_between(x_1, P_choice_mean_1 - P_choice_sem_1, P_choice_mean_1 + P_choice_sem_1,
    #                 color='gray', alpha=0.3)



    #
    plt.ylabel('performance')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.yticks([0, 0.25, 0.5, 0.75, 1])
    plt.ylim([0.1, 1])
    # plt.xlim([0, 45])
    ticks = [0, 10,20,30,40,50]
    plt.xticks(ticks, [int((t-20)*50) for t in ticks])  # Show [0, 5, 10] instead of [0, 50, 100]



    # file_name='combine_'+task+'_'+hp['type']+'_bs'+str(hp['batch_size'])
    # plt.title(file_name, fontsize=8)
    #plt.legend(fontsize=8)
    # fig.savefig(figure_path + file_name +'.pdf')
    plt.show()

switching_sem_combine(task='rdmrt',type='type1')








def plot_diff_uncert_box(type):
    hp['type'] = type
    hp['task'] = 'rdmrt'



    models_0 =[1,2,4,6,8,  10,13,5,18,19]#[1,2,4,5,6,  8,10,13,18,19]# [1,2,4,5,6,   8,10,13,0,16]
    models_1 = [1,2,4,6,8,  10,13,5,18,19]#[1,2,4,5,6,  8,10,13,18,19]


    # iter_list_0=[]
    # iter_list_1 = []
    #
    # for model_idx in np.array(models_0):
    #     _, _, iter_0 = get_log(model_idx=model_idx,context='conB',p_out_uncert=0.8)
    #     iter_list_0.append(iter_0)
    #
    # for model_idx in np.array(models_1):
    #     _, _, iter_1 = get_log(model_idx=model_idx, context='conB', p_out_uncert=0.9)
    #     iter_list_1.append(iter_1)
    #
    #
    # print('iter_list_0:',len(iter_list_0),iter_list_0)
    # print('iter_list_1:', len(iter_list_1),iter_list_1)
    #
    # np.save(data_path + 'iter_list_0_conB.npy', iter_list_0)
    # np.save(data_path + 'iter_list_1_conB.npy', iter_list_1)

    iter_list_0 = list(np.load(data_path + 'iter_list_0_conB.npy'))
    iter_list_1 = list(np.load(data_path + 'iter_list_1_conB.npy'))








    from scipy import stats
    p01 = stats.ranksums(iter_list_0, iter_list_1)
    print('p01', p01)



    all_data =  np.array([iter_list_0, iter_list_1]).T
    print('all_data',all_data)


    fig = plt.figure(figsize=(2., 2.))
    ax = fig.add_axes([0.3, 0.1, 0.55, 0.7])
    colors = sns.color_palette()  # sns.color_palette("Set2")

    labels = ['0.8', '0.9']



    # rectangular box plot
    bplot1 = ax.boxplot(all_data,
                             notch=True,   # vertical box alignment
                             vert=True,
                             patch_artist=True,  # fill with color
                        widths=0.3,
                        showfliers='',
                        labels=labels
                             )  # will be used to label x-ticks
    # fill with colors
    colors = sns.color_palette("Set2")#sns.color_palette("hls", 8)

    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)

    #plt.ylabel('trials', fontsize=12)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # file_name = hp['task'] + '_' +'switching_box_' +  hp['type'] + '_bs' + str(hp['batch_size'])
    #plt.title(file_name, fontsize=5)
    # plt.legend(fontsize=8)
    # fig.savefig(figure_path + file_name + '.pdf')
    plt.show()

plot_diff_uncert_box(type='type1')




def plot_diff_uncert_box_conA(type):
    hp['type'] = type
    hp['task'] = 'rdmrt'


    model_lsit = range(10)#[1,2,4,5,6,   8,10,13,0,16]

    models_0 = model_lsit#range(10)#[1,2,4,5,6,   8,10,13,0,16]
    models_1 = model_lsit#model_lsit#range(10)#[1,2,4,5,6,   8,10,13,0,16]
    models_2 = model_lsit#range(10)#[1,2,4,5,6,   8,10,13,0,16]
    models_3 = model_lsit#[0,1,2,3,4, 5,6,7,17,16]#range(20)#[1,2,4,5,6,   8,10,13,0,16]




    # iter_list_0=[]
    # iter_list_1 = []
    # iter_list_2 = []
    # iter_list_3 = []
    #
    # for model_idx in np.array(models_0):
    #     _, _, iter_0 = get_log(model_idx=model_idx,context='conA',p_out_uncert=0.7)
    #     iter_list_0.append(iter_0)
    #
    # for model_idx in np.array(models_1):
    #     _, _, iter_1 = get_log(model_idx=model_idx,context='conA',p_out_uncert=0.8)
    #     iter_list_1.append(iter_1)
    #
    # for model_idx in np.array(models_2):
    #     _, _, iter_2 = get_log(model_idx=model_idx, context='conA', p_out_uncert=0.9)
    #     iter_list_2.append(iter_2)
    #
    #
    # for model_idx in np.array(models_3):
    #     _, _, iter_3 = get_log(model_idx=model_idx, context='conA', p_out_uncert=1.0)
    #     iter_list_3.append(iter_3)
    #
    # np.save(data_path + 'iter_list_0_conA.npy', iter_list_0)
    # np.save(data_path + 'iter_list_1_conA.npy', iter_list_1)
    # np.save(data_path + 'iter_list_2_conA.npy', iter_list_2)
    # np.save(data_path + 'iter_list_3_conA.npy', iter_list_3)

    iter_list_0 = list(np.load(data_path + 'iter_list_0_conA.npy'))
    iter_list_1 = list(np.load(data_path + 'iter_list_1_conA.npy'))
    iter_list_2 = list(np.load(data_path + 'iter_list_2_conA.npy'))
    iter_list_3 = list(np.load(data_path + 'iter_list_3_conA.npy'))

    print('iter_list_0:',len(iter_list_0),iter_list_0)
    print('iter_list_1:', len(iter_list_1),iter_list_1)
    print('iter_list_2:', len(iter_list_2), iter_list_2)
    print('iter_list_3:', len(iter_list_3), iter_list_3)

    all_data =  np.array([iter_list_0, iter_list_1, iter_list_2,iter_list_3,]).T
    print('all_data',all_data)




    fig = plt.figure(figsize=(3.5, 2.2))
    ax = fig.add_axes([0.3, 0.1, 0.55, 0.7])
    colors = sns.color_palette()  # sns.color_palette("Set2")

    labels = ['0.7','0.8', '0.9', '1.0']


    # rectangular box plot
    bplot1 = ax.boxplot(all_data,
                             notch=True,   # vertical box alignment
                             vert=True,
                             patch_artist=True,  # fill with color
                        widths=0.3,
                        showfliers='',
                        labels=labels
                             )  # will be used to label x-ticks
    # fill with colors
    colors = sns.color_palette("Blues")#sns.color_palette("Set2")#sns.color_palette("hls", 8)

    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)

    #plt.ylabel('trials', fontsize=12)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.yticks([])
    # plt.ylim([1500,2900])


    # file_name = 'conA_'+hp['task'] +'_switching_box_' +  hp['type'] + '_bs' + str(hp['batch_size'])
    #plt.title(file_name, fontsize=5)
    # plt.legend(fontsize=8)
    # fig.savefig(figure_path + file_name + '.pdf')
    plt.show()

plot_diff_uncert_box_conA(type='type1')



























