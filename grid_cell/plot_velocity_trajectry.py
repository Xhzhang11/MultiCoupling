
import sys, os
import numpy as np
import torch


sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"../..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"../")))

from matplotlib import pyplot as plt




import seaborn as sns
import pandas as pd
import scipy.stats as stats
#load parames

hp = {}
hp['root_path'] = os.path.abspath(os.path.join(os.getcwd(),"./"))


hp['Ng']=512

# hp['rng'] = np.random.RandomState(1)


hp['batch_size_test']=512
hp['batch_size']=512
hp['is_EI'] = 'EIno'
hp['wDG'] = 0.4
hp['wc'] = 1.0
hp['wa'] = 0.5
hp['mode'] = 'test'
hp['env'] = 'circle'
hp['activation'] = 'relu'
hp['activation_md'] = 'relu'
hp['activation_CA3'] = 'sigmoid'
hp['type'] = 'type1'
hp['freeze'] = 'no'
hp['sparsity'] = 0.5
hp['init'] = 'rand'
hp['dt'] = 20
hp['Np'] = 512
hp['k'] = 1
hp['learning_rate'] = 0.00005
hp['speed'] = 5
rng = np.random#.RandomState(0)

hp['rng'] = rng

# hp['degree'] = 90#0,30,45,60,90

hp['save_dir'] = hp['root_path']+'/model'
print('save_dir', hp['save_dir'])

fig_path = hp['root_path'] + '/Figures/'
figure_path = os.path.join(fig_path, 'plot_velocity_trajectory' + '/')


data_root = hp['root_path'] + '/Datas/'
data_path = os.path.join(data_root, 'gridcell_dffinput_5.12_k1_circle_even_alpha' + '/')








def plot_pdf_tau(sequence_length):
    sequence_length=sequence_length
    data_path_0 = os.path.join(data_path, 'plot_pdf_tau' + '/')

    # speed_speed0_0 = get_velocity_diff_tau(model_idx=100, sl=sequence_length, tau=25, speed_scale=0.01)
    # speed_speed0_1 = get_velocity_diff_tau(model_idx=100, sl=sequence_length, tau=20, speed_scale=0.01)
    # speed_speed1_0 = get_velocity_diff_tau(model_idx=100, sl=sequence_length, tau=30, speed_scale=1)
    #
    # np.save(data_path_0+'speed_speed0_0'+str(sequence_length)+'.npy',speed_speed0_0)
    # np.save(data_path_0 + 'speed_speed0_1'+str(sequence_length)+'.npy', speed_speed0_1)
    # np.save(data_path_0 + 'speed_speed1_0'+str(sequence_length)+'.npy', speed_speed1_0)

    speed_speed0_0 = list(np.load(data_path_0+'speed_speed0_0'+str(sequence_length)+'.npy'))
    speed_speed0_1 = list(np.load(data_path_0+'speed_speed0_1'+str(sequence_length)+'.npy'))
    speed_speed1_0 = list(np.load(data_path_0 + 'speed_speed1_0'+str(sequence_length)+'.npy'))

    print('speed_speed0_1',np.mean(speed_speed0_1),np.std(speed_speed0_1))


    fig = plt.figure(figsize=(3., 2.))
    ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])
    sns.kdeplot(speed_speed0_0, bw_adjust=0.5, fill=True,label='mental')
    sns.kdeplot(speed_speed1_0, bw_adjust=0.5, fill=True,label='normal')
    ax.set_title('tau=25',fontsize=5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    ax.legend(fontsize=7)
    plt.xlim([-0.05,0.85])
    plt.xlabel('Velocity')
    # plt.savefig(figure_path +  'tau_' + '25.pdf')
    plt.show()




    fig2 = plt.figure(figsize=(3., 2.))
    ax2 = fig2.add_axes([0.15, 0.15, 0.75, 0.75])
    sns.kdeplot(speed_speed0_1, bw_adjust=0.5, fill=True,label='mental')
    sns.kdeplot(speed_speed1_0, bw_adjust=0.5, fill=True,label='normal')
    ax2.set_title('tau=20',fontsize=5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    ax2.legend(fontsize=7)
    plt.xlim([-0.05, 0.85])
    plt.xlabel('Velocity')
    # plt.savefig(figure_path +  'tau_' + '20.pdf')

    plt.show()


plot_pdf_tau(sequence_length=14)



