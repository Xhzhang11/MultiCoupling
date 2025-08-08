import sys,os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
import scipy.stats as stats


import seaborn as sns
from scipy import stats
import json
import pandas as pd

############ model

hp = {}
hp['root_path'] = os.path.abspath(os.path.join(os.getcwd(),"./"))

hp['noise_var'] = 0.1
hp['seed'] = 0
hp['type'] = 'type0'
hp['rnn'] = 128
hp['md'] = 128
hp['batch_size'] = 64
hp['seq_len'] = 20
hp['npoints'] = 500
hp['tau'] = 40
hp['lr'] = 0.0001




fig_path = hp['root_path']+'/Figures/'
figure_path = os.path.join(fig_path, 'speed_plot/')




data_root = hp['root_path']+'/Datas/'
data_path = os.path.join(data_root, 'speed_plot_cere_difforder/')








def get_model_dir_diff_context(model_idx):

    hp['model_idx'] = model_idx

    model_name = hp['type'] + '_n' + str(hp['noise_var']) + '_dealy' + str(hp['delay']) \
                 + '_sl' + str(hp['seq_len']) + '_' + str(hp['npoints']) \
                 + '_' + hp['task'] + '_' + hp['if_cere'] + '_' + str(hp['model_idx'])


    # hp['model_dir_current'] = hp['root_path']+os.path.join('/'+'model_A2B_'+str(hp['p_coh']),  model_name)
    local_folder_name = os.path.join('/' + 'models_saved', model_name)

    model_dir = hp['root_path'] + local_folder_name+'/'
    # print(model_dir)
    #
    if not os.path.exists(model_dir):
        print(f"The path '{model_dir}' does not exist.")
        sys.exit(0)

    return model_dir, model_name



def plot_speed(type,model_idx):
    hp['type'] = type
    hp['if_cere'] = 'yes'

    hp['noise_var'] = 0.3
    hp['delay'] = 200
    hp['seq_len'] = 20
    hp['npoints'] = 8
    hp['task'] = 'line'
    model_dir, model_name = get_model_dir_diff_context(model_idx=model_idx)
    hp['model_dir'] = model_dir

    #print(model_dir)

    with open(model_dir+'/log.json', 'r') as file:
        data = json.load(file)
        epochs = data['epochs'][-1]
        mean_endpoint_error = data['mean_endpoint_error']

    return epochs,mean_endpoint_error




def plot_speed_switching(type,model_idx):
    hp['type']=type
    hp['if_cere']='yes'

    hp['noise_var'] = 0.3
    hp['delay']=200
    hp['seq_len']=20
    hp['npoints']=8
    hp['task']='line'
    model_dir, model_name = get_model_dir_diff_context(model_idx=model_idx)
    hp['model_dir'] = model_dir
    idx_B=0

    #print(model_dir)

    with open(model_dir+'/log_B_'+str(idx_B)+'.json', 'r') as file:
        data = json.load(file)
        epochs = data['epochs'][-1]
        mean_endpoint_error = data['mean_endpoint_error']

    return epochs,mean_endpoint_error







def plot_training_3panel(rule_name):
    print('==========================plot_box')


    # epoch_0_list = []
    # epoch_1_list = []
    # epoch_2_list = []
    #
    # for model_idx in range(1, 60):
    #     epochs_type0, _ = plot_speed(type='type0', model_idx=model_idx)
    #     if 10000<epochs_type0<1000000:
    #     # if model_idx not in np.array([4,13,14,17,48]):
    #         epoch_0_list.append(epochs_type0)
    #
    # for model_idx in range(1, 60):
    #     epochs_type1, _ = plot_speed(type='type1', model_idx=model_idx)
    #     print(model_idx,epochs_type1)
    #     if epochs_type1 < 1000000:
    #
    #     # if model_idx not in np.array([4,13,14,17,48]):
    #         epoch_1_list.append(epochs_type1)
    #
    #
    # for model_idx in range(1, 60):
    #     epochs_type5, _ = plot_speed(type='type5', model_idx=model_idx)
    #     print(model_idx,epochs_type5)
    #     if epochs_type5 < 1000000:
    #
    #     # if model_idx not in np.array([4,13,14,17,48]):
    #         epoch_2_list.append(epochs_type5)
    #
    # epoch_0_list = epoch_0_list[:50]
    # epoch_1_list = epoch_1_list[:50]
    # epoch_2_list = epoch_1_list[:50]
    # print(len(epoch_0_list), '==epoch_0, mean=', np.mean(epoch_0_list), epoch_0_list)
    # print(len(epoch_1_list), '==epoch_1, mean=', np.mean(epoch_1_list), epoch_1_list)
    #
    #
    # np.save(data_path + 'plot_training_3panel_' + 'epoch_0_list.npy', epoch_0_list)
    # np.save(data_path + 'plot_training_3panel_' + 'epoch_1_list.npy', epoch_1_list)
    # np.save(data_path + 'plot_training_3panel_' + 'epoch_2_list.npy', epoch_2_list)
    #







    epoch_0_list = list(np.load(data_path + 'plot_training_3panel_' + 'epoch_0_list.npy'))
    epoch_1_list = list(np.load(data_path + 'plot_training_3panel_' + 'epoch_1_list.npy'))
    epoch_2_list = list(np.load(data_path + 'plot_training_3panel_' + 'epoch_2_list.npy'))





    # Flatten and prepare data
    data = epoch_0_list + epoch_1_list + epoch_2_list
    labels = ['type0'] * len(epoch_0_list) + ['type1'] * len(epoch_1_list) + ['type5'] * len(epoch_2_list)

    df = sns.load_dataset("tips")  # dummy just to avoid errors in IDEs

    fig = plt.figure(figsize=(2.6, 2.0))
    ax = fig.add_axes([0.3, 0.1, 0.55, 0.7])

    box = sns.boxplot(x=labels, y=data, ax=ax,
                      notch=True,

                      palette=[sns.color_palette()[0], sns.color_palette()[1], sns.color_palette()[2]],
                      width=0.3, linewidth=1,
                      showfliers=True)

    # Adjust the size of the outliers (fliers)
    for line in ax.lines:
        if line.get_marker() == 'o':  # circles are outliers
            line.set_markersize(4)  # set desired marker size (e.g., 4)

    ax.set_xlabel("Prob(reward)", fontsize=10)
    ax.set_ylabel("No. epochs", fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate significance
    # Annotate significance
    p01 = stats.ranksums(epoch_0_list, epoch_1_list).pvalue
    p02 = stats.ranksums(epoch_0_list, epoch_2_list).pvalue
    p12 = stats.ranksums(epoch_1_list, epoch_2_list).pvalue
    print('p01', p01)
    print('p02', p02)
    print('p12', p12)
    # annotator = Annotator(ax, pairs, data=pd.DataFrame({"x": labels, "y": data}), x="x", y="y")
    # annotator.configure(test=None, text_format='star', loc='outside', verbose=0)
    # annotator.set_custom_annotations([f"p = {p_val:.3f}"])
    # annotator.annotate()
    plt.title(str(p01)+'\n'+str(p02)+'\n'+str(p12),fontsize=8)

    plt.tight_layout()
    # fig.savefig(figure_path + 'speed_training_three_type.pdf')
    plt.show()
plot_training_3panel('line')




def plot_notched_box(rule_name):
    print('==========================plot_box')

    # epoch_0_list = []
    # epoch_1_list = []
    #
    # for model_idx in range(1, 60):
    #     epochs_type0, _ = plot_speed_switching(type='type1', model_idx=model_idx)
    #     if epochs_type0<1000:
    #     # if model_idx not in np.array([4,13,14,17,48]):
    #         epoch_0_list.append(epochs_type0)
    #
    # for model_idx in range(1, 60):
    #     epochs_type1, _ = plot_speed_switching(type='type5', model_idx=model_idx)
    #     print(model_idx,epochs_type1)
    #     if epochs_type1 < 1000:
    #
    #     # if model_idx not in np.array([4,13,14,17,48]):
    #         epoch_1_list.append(epochs_type1)
    #
    # epoch_0_list = epoch_0_list[:50]
    # epoch_1_list = epoch_1_list[:50]
    # print(len(epoch_0_list), '==epoch_0, mean=', np.mean(epoch_0_list), epoch_0_list)
    # print(len(epoch_1_list), '==epoch_1, mean=', np.mean(epoch_1_list), epoch_1_list)
    #
    # np.save(data_path + 'plot_notched_box_' + 'epoch_0_list.npy', epoch_0_list)
    # np.save(data_path + 'plot_notched_box_' + 'epoch_1_list.npy', epoch_1_list)






    epoch_0_list = list(np.load(data_path + 'plot_notched_box_' + 'epoch_0_list.npy'))
    epoch_1_list = list(np.load(data_path + 'plot_notched_box_' + 'epoch_1_list.npy'))


    # Flatten and prepare data
    data = epoch_0_list + epoch_1_list
    labels = ['no'] * len(epoch_0_list) + ['cere'] * len(epoch_1_list)

    df = sns.load_dataset("tips")  # dummy just to avoid errors in IDEs

    fig = plt.figure(figsize=(3, 2.2))
    ax = fig.add_axes([0.3, 0.1, 0.55, 0.7])

    box = sns.boxplot(x=labels, y=data, ax=ax,
                      notch=True,

                      palette=[sns.color_palette()[0], sns.color_palette()[1]],
                      width=0.3, linewidth=1,
                      showfliers=True)

    # Adjust the size of the outliers (fliers)
    for line in ax.lines:
        if line.get_marker() == 'o':  # circles are outliers
            line.set_markersize(4)  # set desired marker size (e.g., 4)

    ax.set_xlabel("Prob(reward)", fontsize=10)
    ax.set_ylabel("No. epochs", fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate significance
    p01 = stats.ranksums(epoch_0_list, epoch_1_list).pvalue

    print('p01', p01)

    pairs = [("0.8", "0.9")]
    # annotator = Annotator(ax, pairs, data=pd.DataFrame({"x": labels, "y": data}), x="x", y="y")
    # annotator.configure(test=None, text_format='star', loc='outside', verbose=0)
    # annotator.set_custom_annotations([f"p = {p_val:.3f}"])
    # annotator.annotate()
    plt.title(str(p01),fontsize=8)

    plt.tight_layout()
    # fig.savefig(figure_path + 'speed_cere_without.pdf')
    plt.show()
plot_notched_box('line')





def plot_notched_box_type0(rule_name):
    print('==========================plot_box')

    # epoch_0_list = []
    # epoch_1_list = []
    #
    # for model_idx in range(1, 60):
    #     epochs_type0, _ = plot_speed_switching(type='type0', model_idx=model_idx)
    #     if epochs_type0<1000:
    #         epoch_0_list.append(epochs_type0)
    #
    # for model_idx in range(0, 60):
    #     epochs_type1, _ = plot_speed_switching(type='type1', model_idx=model_idx)
    #     # print(model_idx,epochs_type1)
    #     # if model_idx not in np.array([22]):
    #     if epochs_type1 < 1000:
    #         epoch_1_list.append(epochs_type1)
    #
    # epoch_0_list = epoch_0_list[:50]
    # epoch_1_list = epoch_1_list[:50]
    # print(len(epoch_0_list), '==epoch_0, mean=', np.mean(epoch_0_list), epoch_0_list)
    # print(len(epoch_1_list), '==epoch_1, mean=', np.mean(epoch_1_list), epoch_1_list)
    #
    # np.save(data_path + 'plot_notched_box_type0_' + 'epoch_0_list.npy', epoch_0_list)
    # np.save(data_path + 'plot_notched_box_type0_' + 'epoch_1_list.npy', epoch_1_list)
    #







    epoch_0_list = list(np.load(data_path + 'plot_notched_box_type0_' + 'epoch_0_list.npy'))
    epoch_1_list = list(np.load(data_path + 'plot_notched_box_type0_' + 'epoch_1_list.npy'))










    # Flatten and prepare data
    data = epoch_0_list + epoch_1_list
    labels = ['type0'] * len(epoch_0_list) + ['type1'] * len(epoch_1_list)

    df = sns.load_dataset("tips")  # dummy just to avoid errors in IDEs

    fig = plt.figure(figsize=(2.2, 2.2))
    ax = fig.add_axes([0.3, 0.1, 0.55, 0.7])

    box = sns.boxplot(x=labels, y=data, ax=ax,
                      notch=True,

                      palette=[sns.color_palette()[0], sns.color_palette()[1]],
                      width=0.3, linewidth=1,
                      showfliers=True)

    # Adjust the size of the outliers (fliers)
    for line in ax.lines:
        if line.get_marker() == 'o':  # circles are outliers
            line.set_markersize(4)  # set desired marker size (e.g., 4)

    ax.set_xlabel("Prob(reward)", fontsize=10)
    ax.set_ylabel("No. epochs", fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate significance
    p_val = stats.ranksums(epoch_0_list, epoch_1_list).pvalue
    print('p_val=',str(p_val))
    pairs = [("0.8", "0.9")]
    # annotator = Annotator(ax, pairs, data=pd.DataFrame({"x": labels, "y": data}), x="x", y="y")
    # annotator.configure(test=None, text_format='star', loc='outside', verbose=0)
    # annotator.set_custom_annotations([f"p = {p_val:.3f}"])
    # annotator.annotate()
    plt.title(str(p_val),fontsize=8)

    plt.tight_layout()
    # fig.savefig(figure_path + 'speed_type0VStype1.pdf')
    plt.show()
plot_notched_box_type0('line')