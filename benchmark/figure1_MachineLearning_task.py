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



def plot_valid_accs_tau2_sem_CIFAR(tau):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns


    data_path = os.path.join(data_root, 'CIFAR_accs/')


    #
    # valid_accs_type0_0_list = []
    # valid_accs_type1_0_list = []
    #
    # for seed in np.array([0,1,3,5,6, 7,8,9,11,12]):#0,1,3,5,6,7,8,2,4
    #     valid_accs_type0_0 = get_valid_accs(tau=tau, type='type0', dropout_rate=0.0, batch_size=128, seed=seed)
    #     valid_accs_type1_0 = get_valid_accs(tau=tau, type='type1', dropout_rate=0.0, batch_size=128, seed=seed)
    #
    #     valid_accs_type0_0_list.append(valid_accs_type0_0)
    #     valid_accs_type1_0_list.append(valid_accs_type1_0)
    #
    #     print('valid_accs_type0_0',len(valid_accs_type0_0))
    #     print('valid_accs_type1_0', len(valid_accs_type1_0))
    #
    # # Convert to numpy arrays
    # valid_accs_type0_0_numpy = np.array(valid_accs_type0_0_list)
    # valid_accs_type1_0_numpy = np.array(valid_accs_type1_0_list)
    #
    #
    # np.save(data_path+'valid_accs_type0_0_numpy.npy',valid_accs_type0_0_numpy)
    # np.save(data_path+'valid_accs_type1_0_numpy.npy',valid_accs_type1_0_numpy)

    valid_accs_type0_0_numpy = np.load(data_path + 'valid_accs_type0_0_numpy.npy')
    valid_accs_type1_0_numpy = np.load(data_path + 'valid_accs_type1_0_numpy.npy')








    # Compute mean and SEM
    accs_type0_0_mean = valid_accs_type0_0_numpy.mean(axis=0)
    accs_type1_0_mean = valid_accs_type1_0_numpy.mean(axis=0)
    accs_type0_0_sem = valid_accs_type0_0_numpy.std(axis=0, ddof=1)# / np.sqrt(valid_accs_type0_0_numpy.shape[0])
    accs_type1_0_sem = valid_accs_type1_0_numpy.std(axis=0, ddof=1)# / np.sqrt(valid_accs_type1_0_numpy.shape[0])

    # Plot with shaded error
    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
    x = np.arange(len(accs_type0_0_mean))

    colors = sns.color_palette()

    # Plot means and SEM shaded area
    ax.plot(x, accs_type0_0_mean, c='tab:blue', label='type0')
    ax.fill_between(x, accs_type0_0_mean - accs_type0_0_sem, accs_type0_0_mean + accs_type0_0_sem,
                    color='tab:blue', alpha=0.3)

    ax.plot(x, accs_type1_0_mean, c='tab:red', label='type1')
    ax.fill_between(x, accs_type1_0_mean - accs_type1_0_sem, accs_type1_0_mean + accs_type1_0_sem,
                    color='tab:red', alpha=0.3)

    # Reference lines
    #ax.axhline(y=0.83, color="grey", linestyle="--")
    #ax.axhline(y=0.89, color="grey", linestyle="--")

    # Final touches
    #ax.legend(fontsize=10)
    ax.set_ylim([0.6, 0.85])
    ax.set_title('tau=' + str(tau))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    # plt.savefig(fig_path + 'mean_sem_shade_acc.pdf')
    plt.show()
plot_valid_accs_tau2_sem_CIFAR(tau=1)





def plot_train_loss_tau2_sem_CIFAR(tau):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns


    data_path = os.path.join(data_root, 'CIFAR_loss/')


    # valid_accs_type0_0_list = []
    # valid_accs_type1_0_list = []
    #
    # for seed in np.array([0,1,3,5,6, 7,8,9,11,12]):#0,1,3,5,6,7,8,2,4
    #     valid_accs_type0_0 = get_train_loss(tau=tau, type='type0', dropout_rate=0.0, batch_size=128, seed=seed)
    #     valid_accs_type1_0 = get_train_loss(tau=tau, type='type1', dropout_rate=0.0, batch_size=128, seed=seed)
    #
    #     valid_accs_type0_0_list.append(valid_accs_type0_0)
    #     valid_accs_type1_0_list.append(valid_accs_type1_0)
    # # Convert to numpy arrays
    # valid_accs_type0_0_numpy = np.array(valid_accs_type0_0_list)
    # valid_accs_type1_0_numpy = np.array(valid_accs_type1_0_list)
    #
    # np.save(data_path + 'valid_accs_type0_0_numpy.npy', valid_accs_type0_0_numpy)
    # np.save(data_path + 'valid_accs_type1_0_numpy.npy', valid_accs_type1_0_numpy)

    valid_accs_type0_0_numpy = np.load(data_path + 'valid_accs_type0_0_numpy.npy')
    valid_accs_type1_0_numpy = np.load(data_path + 'valid_accs_type1_0_numpy.npy')

    # Compute mean and SEM
    accs_type0_0_mean = valid_accs_type0_0_numpy.mean(axis=0)
    accs_type1_0_mean = valid_accs_type1_0_numpy.mean(axis=0)
    accs_type0_0_sem = valid_accs_type0_0_numpy.std(axis=0, ddof=1)# / np.sqrt(valid_accs_type0_0_numpy.shape[0])
    accs_type1_0_sem = valid_accs_type1_0_numpy.std(axis=0, ddof=1)# / np.sqrt(valid_accs_type1_0_numpy.shape[0])

    # Plot with shaded error
    fig = plt.figure(figsize=(3, 2.2))
    ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
    x = np.arange(len(accs_type0_0_mean))

    colors = sns.color_palette()

    # Plot means and SEM shaded area
    ax.plot(x, accs_type0_0_mean, c='tab:blue', label='type0')
    ax.fill_between(x, accs_type0_0_mean - accs_type0_0_sem, accs_type0_0_mean + accs_type0_0_sem,
                    color='tab:blue', alpha=0.3)

    ax.plot(x, accs_type1_0_mean, c='tab:red', label='type1')
    ax.fill_between(x, accs_type1_0_mean - accs_type1_0_sem, accs_type1_0_mean + accs_type1_0_sem,
                    color='tab:red', alpha=0.3)

    # Reference lines

    # Final touches
    #ax.legend(fontsize=10)
    # ax.set_ylim([0.2, 0.9])
    ax.set_title('train_loss')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    plt.tight_layout()
    # plt.savefig(fig_path + 'mean_sem_train_loss.pdf')
    plt.show()

plot_train_loss_tau2_sem_CIFAR(tau=1)










def plot_valid_accs_tau2_sem_TREC(tau):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    data_path = os.path.join(data_root, 'TREC_accs/')






    # valid_accs_type0_0_list = []
    # valid_accs_type1_0_list = []
    #
    # for seed in range(1, 11):
    #     valid_accs_type0_0 = get_valid_accs(tau=tau, type='type0', dropout_rate=0.3, batch_size=32, seed=seed)
    #     valid_accs_type1_0 = get_valid_accs(tau=tau, type='type1', dropout_rate=0.3, batch_size=32, seed=seed)
    #
    #     print('valid_accs_type0_0',len(valid_accs_type0_0))
    #     print('valid_accs_type0_1', len(valid_accs_type1_0))
    #     valid_accs_type0_0_list.append(valid_accs_type0_0)
    #     valid_accs_type1_0_list.append(valid_accs_type1_0)
    #
    # print('valid_accs_type0_0_list', valid_accs_type0_0_list)
    # print('valid_accs_type1_0_list', valid_accs_type1_0_list)
    #
    # # Convert to numpy arrays
    # valid_accs_type0_0_numpy = np.array(valid_accs_type0_0_list)
    # valid_accs_type1_0_numpy = np.array(valid_accs_type1_0_list)
    #
    # np.save(data_path + 'valid_accs_type0_0_numpy.npy', valid_accs_type0_0_numpy)
    # np.save(data_path + 'valid_accs_type1_0_numpy.npy', valid_accs_type1_0_numpy)







    valid_accs_type0_0_numpy = np.load(data_path + 'valid_accs_type0_0_numpy.npy')
    valid_accs_type1_0_numpy = np.load(data_path + 'valid_accs_type1_0_numpy.npy')





    # Compute mean and SEM
    accs_type0_0_mean = valid_accs_type0_0_numpy.mean(axis=0)
    accs_type1_0_mean = valid_accs_type1_0_numpy.mean(axis=0)
    accs_type0_0_sem = valid_accs_type0_0_numpy.std(axis=0, ddof=1)# / np.sqrt(valid_accs_type0_0_numpy.shape[0])
    accs_type1_0_sem = valid_accs_type1_0_numpy.std(axis=0, ddof=1)# / np.sqrt(valid_accs_type1_0_numpy.shape[0])

    # Plot with shaded error
    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
    x = np.arange(len(accs_type0_0_mean))

    colors = sns.color_palette()

    # Plot means and SEM shaded area
    ax.plot(x, accs_type0_0_mean, c='tab:blue', label='type0')
    ax.fill_between(x, accs_type0_0_mean - accs_type0_0_sem, accs_type0_0_mean + accs_type0_0_sem,
                    color='tab:blue', alpha=0.3)

    ax.plot(x, accs_type1_0_mean, c='tab:red', label='type1')
    ax.fill_between(x, accs_type1_0_mean - accs_type1_0_sem, accs_type1_0_mean + accs_type1_0_sem,
                    color='tab:red', alpha=0.3)

    # Reference lines
    #ax.axhline(y=0.86, color="grey", linestyle="--")
    #ax.axhline(y=0.89, color="grey", linestyle="--")

    # Final touches
    # ax.legend(fontsize=10)
    ax.set_ylim([0.2, 0.9])
    ax.set_title('tau=' + str(tau))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    # plt.savefig(fig_path + 'mean_sem_shade_acc.pdf')
    plt.show()


plot_valid_accs_tau2_sem_TREC(tau=1)




def plot_train_loss_tau2_sem_TREC(tau):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    data_path = os.path.join(data_root, 'TREC_loss/')

    # valid_accs_type0_0_list = []
    # valid_accs_type1_0_list = []
    #
    # for seed in range(1, 11):
    #     valid_accs_type0_0 = get_train_loss(tau=tau, type='type0', dropout_rate=0.3, batch_size=32, seed=seed)
    #     valid_accs_type1_0 = get_train_loss(tau=tau, type='type1', dropout_rate=0.3, batch_size=32, seed=seed)
    #
    #     print('valid_accs_type0_0',len(valid_accs_type0_0))
    #     print('valid_accs_type0_1', len(valid_accs_type1_0))
    #     valid_accs_type0_0_list.append(valid_accs_type0_0)
    #     valid_accs_type1_0_list.append(valid_accs_type1_0)
    #
    # print('valid_accs_type0_0_list', valid_accs_type0_0_list)
    # print('valid_accs_type1_0_list', valid_accs_type1_0_list)
    #
    #
    #
    #
    # # Convert to numpy arrays
    # valid_accs_type0_0_numpy = np.array(valid_accs_type0_0_list)
    # valid_accs_type1_0_numpy = np.array(valid_accs_type1_0_list)
    #
    # np.save(data_path + 'valid_accs_type0_0_numpy.npy', valid_accs_type0_0_numpy)
    # np.save(data_path + 'valid_accs_type1_0_numpy.npy', valid_accs_type1_0_numpy)




    valid_accs_type0_0_numpy = np.load(data_path + 'valid_accs_type0_0_numpy.npy')
    valid_accs_type1_0_numpy = np.load(data_path + 'valid_accs_type1_0_numpy.npy')

    # Compute mean and SEM
    accs_type0_0_mean = valid_accs_type0_0_numpy.mean(axis=0)
    accs_type1_0_mean = valid_accs_type1_0_numpy.mean(axis=0)
    accs_type0_0_sem = valid_accs_type0_0_numpy.std(axis=0, ddof=1)# / np.sqrt(valid_accs_type0_0_numpy.shape[0])
    accs_type1_0_sem = valid_accs_type1_0_numpy.std(axis=0, ddof=1)# / np.sqrt(valid_accs_type1_0_numpy.shape[0])

    # Plot with shaded error
    fig = plt.figure(figsize=(3, 2.2))
    ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
    x = np.arange(len(accs_type0_0_mean))

    colors = sns.color_palette()

    # Plot means and SEM shaded area
    ax.plot(x, accs_type0_0_mean, c='tab:blue', label='type0')
    ax.fill_between(x, accs_type0_0_mean - accs_type0_0_sem, accs_type0_0_mean + accs_type0_0_sem,
                    color='tab:blue', alpha=0.3)

    ax.plot(x, accs_type1_0_mean, c='tab:red', label='type1')
    ax.fill_between(x, accs_type1_0_mean - accs_type1_0_sem, accs_type1_0_mean + accs_type1_0_sem,
                    color='tab:red', alpha=0.3)

    # Reference lines
    #ax.axhline(y=0.86, color="grey", linestyle="--")
    #ax.axhline(y=0.89, color="grey", linestyle="--")

    # Final touches
    #ax.legend(fontsize=10)
    # ax.set_ylim([0.2, 0.9])
    #ax.set_title('training loss')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")

    plt.tight_layout()
    # plt.savefig(fig_path + 'training loss.pdf')
    plt.show()


plot_train_loss_tau2_sem_TREC(tau=1)














def plot_valid_accs_tau2_sem_IMDB(tau):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns


    data_path = os.path.join(data_root, 'IMDB_accs/')


    # valid_accs_type0_0_list = []
    # valid_accs_type1_0_list = []
    #
    # for seed in range(1, 11):
    #     valid_accs_type0_0 = get_valid_accs(tau=tau, type='type0', dropout_rate=0.0, batch_size=256, seed=seed)
    #     valid_accs_type1_0 = get_valid_accs(tau=tau, type='type1', dropout_rate=0.0, batch_size=256, seed=seed)
    #
    #     valid_accs_type0_0_list.append(valid_accs_type0_0)
    #     valid_accs_type1_0_list.append(valid_accs_type1_0)
    #
    # # Convert to numpy arrays
    # valid_accs_type0_0_numpy = np.array(valid_accs_type0_0_list)
    # valid_accs_type1_0_numpy = np.array(valid_accs_type1_0_list)
    #
    # np.save(data_path + 'valid_accs_type0_0_numpy.npy', valid_accs_type0_0_numpy)
    # np.save(data_path + 'valid_accs_type1_0_numpy.npy', valid_accs_type1_0_numpy)

    valid_accs_type0_0_numpy = np.load(data_path + 'valid_accs_type0_0_numpy.npy')
    valid_accs_type1_0_numpy = np.load(data_path + 'valid_accs_type1_0_numpy.npy')








    # Compute mean and SEM
    accs_type0_0_mean = valid_accs_type0_0_numpy.mean(axis=0)
    accs_type1_0_mean = valid_accs_type1_0_numpy.mean(axis=0)
    accs_type0_0_sem = valid_accs_type0_0_numpy.std(axis=0, ddof=1)# / np.sqrt(valid_accs_type0_0_numpy.shape[0])
    accs_type1_0_sem = valid_accs_type1_0_numpy.std(axis=0, ddof=1)# / np.sqrt(valid_accs_type1_0_numpy.shape[0])

    # Plot with shaded error
    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
    x = np.arange(len(accs_type0_0_mean))

    colors = sns.color_palette()

    # Plot means and SEM shaded area
    ax.plot(x, accs_type0_0_mean, c='tab:blue', label='type0')
    ax.fill_between(x, accs_type0_0_mean - accs_type0_0_sem, accs_type0_0_mean + accs_type0_0_sem,
                    color='tab:blue', alpha=0.3)

    ax.plot(x, accs_type1_0_mean, c='tab:red', label='type1')
    ax.fill_between(x, accs_type1_0_mean - accs_type1_0_sem, accs_type1_0_mean + accs_type1_0_sem,
                    color='tab:red', alpha=0.3)

    # Reference lines
    # ax.axhline(y=0.85, color="blue", linestyle="--")
    #ax.axhline(y=0.86, color="grey", linestyle="--")

    # Final touches
    #ax.legend(fontsize=10)
    ax.set_ylim([0.5, 0.92])
    ax.set_title('tau=' + str(tau))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    # plt.savefig(fig_path + 'mean_sem_shade_acc.pdf')
    plt.show()


plot_valid_accs_tau2_sem_IMDB(tau=2.0)





def plot_train_loss_tau2_sem_IMDB(tau):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    data_path = os.path.join(data_root, 'IMDB_loss/')






    # valid_accs_type0_0_list = []
    # valid_accs_type1_0_list = []
    #
    # for seed in range(1, 11):
    #     valid_accs_type0_0 = get_train_loss(tau=tau, type='type0', dropout_rate=0.0, batch_size=256, seed=seed)
    #     valid_accs_type1_0 = get_train_loss(tau=tau, type='type1', dropout_rate=0.0, batch_size=256, seed=seed)
    #
    #     valid_accs_type0_0_list.append(valid_accs_type0_0)
    #     valid_accs_type1_0_list.append(valid_accs_type1_0)
    #
    # # Convert to numpy arrays
    # valid_accs_type0_0_numpy = np.array(valid_accs_type0_0_list)
    # valid_accs_type1_0_numpy = np.array(valid_accs_type1_0_list)
    #
    # np.save(data_path + 'valid_accs_type0_0_numpy.npy', valid_accs_type0_0_numpy)
    # np.save(data_path + 'valid_accs_type1_0_numpy.npy', valid_accs_type1_0_numpy)

    valid_accs_type0_0_numpy = np.load(data_path + 'valid_accs_type0_0_numpy.npy')
    valid_accs_type1_0_numpy = np.load(data_path + 'valid_accs_type1_0_numpy.npy')










    # Compute mean and SEM
    accs_type0_0_mean = valid_accs_type0_0_numpy.mean(axis=0)
    accs_type1_0_mean = valid_accs_type1_0_numpy.mean(axis=0)
    accs_type0_0_sem = valid_accs_type0_0_numpy.std(axis=0, ddof=1)# / np.sqrt(valid_accs_type0_0_numpy.shape[0])
    accs_type1_0_sem = valid_accs_type1_0_numpy.std(axis=0, ddof=1)# / np.sqrt(valid_accs_type1_0_numpy.shape[0])

    # Plot with shaded error
    fig = plt.figure(figsize=(3, 2.2))
    ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
    x = np.arange(len(accs_type0_0_mean))

    colors = sns.color_palette()

    # Plot means and SEM shaded area
    ax.plot(x, accs_type0_0_mean, c='tab:blue', label='type0')
    ax.fill_between(x, accs_type0_0_mean - accs_type0_0_sem, accs_type0_0_mean + accs_type0_0_sem,
                    color='tab:blue', alpha=0.3)

    ax.plot(x, accs_type1_0_mean, c='tab:red', label='type1')
    ax.fill_between(x, accs_type1_0_mean - accs_type1_0_sem, accs_type1_0_mean + accs_type1_0_sem,
                    color='tab:red', alpha=0.3)


    # Final touches
    #ax.legend(fontsize=10)
    # ax.set_ylim([0.5, 0.9])
    #ax.set_title('train_loss')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    plt.tight_layout()
    # plt.savefig(fig_path + 'plot_train_loss_tau2_sem.pdf')
    plt.show()


plot_train_loss_tau2_sem_IMDB(tau=2.0)
































