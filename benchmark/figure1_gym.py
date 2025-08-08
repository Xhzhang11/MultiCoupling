
import os
import torch
import argparse
import matplotlib.pyplot as plt


import numpy as np
import seaborn as sns


device = torch.device("cpu")
torch.manual_seed(1)
root_path = os.path.abspath(os.path.join(os.getcwd(),"./"))
hp= {}
data_root = os.path.join(root_path + '/' + 'Datas/')








def plot_statistic_result_histogram_CartPole():
    data_path = os.path.join(data_root, 'CartPole/')

    # N_model=30
    # total_time_list_type0 = []
    # total_time_list_type1 = []
    #
    #
    # for model_idx in range(0,N_model):
    #     # print(model_idx)
    #     total_time_type0 = plot_run_time(model_idx,type='type0')
    #     total_time_list_type0.append(total_time_type0)
    #
    # for model_idx in range(0,N_model):
    #     # print(model_idx)
    #     total_time_type1= plot_run_time(model_idx, type='type1')
    #     total_time_list_type1.append(total_time_type1)
    #
    # np.save(data_path+'total_time_list_type0.npy',total_time_list_type0)
    # np.save(data_path+'total_time_list_type1.npy',total_time_list_type1)


    total_time_list_type0 = np.load(data_path + 'total_time_list_type0.npy')
    total_time_list_type1 = np.load(data_path + 'total_time_list_type1.npy')



    total_time_list_type0_norm = total_time_list_type0/np.mean(total_time_list_type0)
    total_time_list_type1_norm = total_time_list_type1/np.mean(total_time_list_type0)

    IT_0 = np.mean(total_time_list_type0_norm)
    IT_1 = np.mean(total_time_list_type1_norm)

    IT_std_0 = np.std(total_time_list_type0_norm) / np.sqrt(len(total_time_list_type0_norm))
    IT_std_1 = np.std(total_time_list_type1_norm) / np.sqrt(len(total_time_list_type1_norm))


    IT_mean = [IT_0, IT_1]
    IT_std = [IT_std_0, IT_std_1]

    from scipy.stats import ranksums
    p = ranksums(total_time_list_type0, total_time_list_type1)
    print('p', p)

    fig = plt.figure(figsize=(2.2, 2.2))
    ax = fig.add_axes([0.3, 0.1, 0.55, 0.7])

    # fig = plt.figure(figsize=(2, 2))
    # ax = fig.add_axes([0.4, 0.1, 0.55, 0.7])
    br1 = 1 * np.arange(len(IT_mean))

    colors = sns.color_palette("Set2")

    # ax.bar(name_context,IT_mean, yerr =IT_std,color =['black','tab:blue','r'], width=0.3)
    plt.bar('Add.', IT_0, yerr=IT_std_0, label='1.0', color='tab:blue', width=0.3, alpha=0.9)
    plt.bar('Multi', IT_1, yerr=IT_std_1, label='0.8', color='tab:orange', width=0.3, alpha=0.9)

    plt.ylabel('iteration', fontsize=10)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title('cartpole'+'\n'+str(p[1]), fontsize=8)
    # plt.yticks([0, 2, 4, 6], fontsize=13)
    plt.xlim([-0.3, 1.3])

    # plt.yticks([0, 2, 4, 6], fontsize=13)

    # plt.legend(fontsize=8)

    # fig.savefig(figure_path + '/speed_cartpole.pdf')
    plt.show()

plot_statistic_result_histogram_CartPole()








def plot_statistic_result_histogram_MountainCar():
    data_path = os.path.join(data_root, 'MountainCar/')

    N_model=15

    # total_time_list_type0 = []
    # total_time_list_type1 = []
    #
    #
    # for model_idx in range(0,N_model):
    #     # print(model_idx)
    #     step,total_time_type0,task_solve = plot_run_time(model_idx,type='type0')
    #
    #     total_time_list_type0.append(total_time_type0)
    #
    #
    # for model_idx in range(0,N_model):
    #     # print(model_idx)
    #     step, total_time_type1, task_solve = plot_run_time(model_idx, type='type1')
    #     total_time_list_type1.append(total_time_type1)
    #
    # np.save(data_path+'total_time_list_type0.npy',total_time_list_type0)
    # np.save(data_path+'total_time_list_type1.npy',total_time_list_type1)

    total_time_list_type0 = np.load(data_path + 'total_time_list_type0.npy')
    total_time_list_type1 = np.load(data_path + 'total_time_list_type1.npy')

    total_time_list_type0_norm = total_time_list_type0/np.mean(total_time_list_type0)
    total_time_list_type1_norm = total_time_list_type1/np.mean(total_time_list_type0)

    IT_0 = np.mean(total_time_list_type0_norm)
    IT_1 = np.mean(total_time_list_type1_norm)

    IT_std_0 = np.std(total_time_list_type0_norm) / np.sqrt(len(total_time_list_type0_norm))
    IT_std_1 = np.std(total_time_list_type1_norm) / np.sqrt(len(total_time_list_type1_norm))


    IT_mean = [IT_0, IT_1]
    IT_std = [IT_std_0, IT_std_1]

    from scipy.stats import ranksums
    p = ranksums(total_time_list_type0, total_time_list_type1)
    print('p', p)

    fig = plt.figure(figsize=(2.2, 2.2))
    ax = fig.add_axes([0.3, 0.1, 0.55, 0.7])


    br1 = 1 * np.arange(len(IT_mean))

    colors = sns.color_palette("Set2")

    # ax.bar(name_context,IT_mean, yerr =IT_std,color =['black','tab:blue','r'], width=0.3)
    plt.bar('Add.', IT_0, yerr=IT_std_0, label='1.0', color='tab:blue', width=0.3, alpha=0.9)
    plt.bar('Multi', IT_1, yerr=IT_std_1, label='0.8', color='tab:orange', width=0.3, alpha=0.9)

    plt.ylabel('iteration', fontsize=10)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title('mountain_car'+'\n'+str(p[1]), fontsize=8)
    # plt.yticks([0, 2, 4, 6], fontsize=13)
    plt.xlim([-0.3, 1.3])

    # plt.yticks([0, 2, 4, 6], fontsize=13)

    # plt.legend(fontsize=8)
    # fig.savefig(figure_path + '/speed_mountaincar.pdf')
    plt.show()




plot_statistic_result_histogram_MountainCar()

































