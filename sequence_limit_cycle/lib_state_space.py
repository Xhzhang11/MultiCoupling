"""
This file contains functions that test the behavior of the model
These functions generally involve some psychometric measurements of the model,
for example performance in decision-making tasks as a function of input strength

These measurements are important as they show whether the network exhibits
some critically important computations, including integration and generalization.
"""


from __future__ import division

import os,sys
import seaborn as sns
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import run
import tools
import torch
import pdb
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA

def get_epoch(hp):
    # print('************ perform HL_task')
    dt = hp['dt']
    batch_size = 1
    rng = hp['rng']
    hp['stim_delay'] = 0



    fix_on = (rng.uniform(0, 0, batch_size) / dt).astype(int)
    cue_on = (rng.uniform(40, 40, batch_size) / dt).astype(int)
    cue_duration = int(hp['cue_duration'] / dt)
    cue_off = cue_on + cue_duration
    cue_delay = int(hp['cue_delay'] / dt)




    stim2_on = cue_off + cue_delay
    stim_duration = int(hp['stim_duration'] / dt)
    stim2_off = stim2_on + stim_duration

    stim_delay = int(hp['stim_delay'] / dt)
    response_on = stim2_off + stim_delay
    response_duration = int(100 / dt)

    # response end time
    response_off = response_on + response_duration


    epoch = {'cue_on': cue_on[0],
             'cue_off': cue_off[0],
             'stim2_on':stim2_on[0],
             'stim2_off':stim2_off[0],

             'response_on':response_on[0],
             'response_off':response_off[0]}
    print(epoch)

    return epoch


def generate_test_trial(context_name,hp,model_dir,
                        c,
                        gamma_bar,

                        batch_size=1):

    rng  = hp['rng']

    runnerObj = run.Runner(rule_name=context_name, hp=hp,model_dir=model_dir, is_cuda=False, noise_on=True,mode='pca')

    trial_input, run_result = runnerObj.run(batch_size=batch_size,
                                            c=c,
                                            gamma_bar=gamma_bar,
                                     )


    return trial_input, run_result

def get_neurons_activity_mode_test1(context_name,hp,type,model_dir,gamma_bar,choice,batch_size):
    hp['type']=type

    runnerObj = run.Runner(rule_name=context_name, hp=hp, model_dir=model_dir, is_cuda=False, noise_on=False,mode='test1')
    trial, run_result = runnerObj.run(batch_size=batch_size, gamma_bar=gamma_bar,choice=choice)
    epochs = trial['epochs']
    #print(trial.x.shape,trial.x[:,0,:])
    #### average value over batch_sizes for hidden state
    firing_rate = run_result.firing_rate_binder.detach().cpu().numpy()
    print('firing_rate',firing_rate.shape)
    firing_rate_list = list([firing_rate[:,i,:] for i in range(batch_size)])
    print('==firing_rate', np.array(firing_rate_list).shape)
    firing_rate_mean = np.mean(np.array(firing_rate_list),axis=0)
    print('firing_rate_mean',firing_rate_mean.shape)


    #### MD
    firing_rate_md = run_result.firing_rate_md.detach().cpu().numpy()
    fr_MD_list = list([firing_rate_md[:,i,:] for i in range(batch_size)])
    fr_MD_mean = np.mean(np.array(fr_MD_list),axis=0)
    fr_MD = fr_MD_mean

    #
    Vt_list = [Vt.detach().cpu().numpy() for Vt in run_result.V_t]
    print('Vt_list',np.array(Vt_list).shape)
    fr_Vt_mean = np.mean(np.array(Vt_list), axis=1)
    fr_Vt = fr_Vt_mean
    print('fr_Vt_mean', fr_Vt_mean.shape)


    return firing_rate_mean,fr_MD,fr_Vt,epochs


def PCA_activity_2D(figure_path,model_name,model_dir,idx,hp):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """

    start_time =0
    end_time=30
    print('plot_activity_diff_tau_6panel',hp['type'])
    rng = hp['rng']
    batch_size=hp['batch_size']
    #c1 = 0.005 * rng.choice([ 6.4, 12.8, 25.6, 51.2], (batch_size,))

    c1 = 0.005 * rng.choice([ 51.2], (batch_size,))
    c2 = 0.005 * rng.choice([ -51.2], (batch_size,))
    print('c',c1)

    trial_input_0, run_result_0 = generate_test_trial(context_name=hp['rule_name'],hp=hp,
                                                            model_dir=model_dir,
                                                            c=c1,gamma_bar=0.5,batch_size=batch_size)

    trial_input_1, run_result_1 = generate_test_trial(context_name=hp['rule_name'], hp=hp,
                                                            model_dir=model_dir,
                                                            c=c2,gamma_bar=0.5,  batch_size=batch_size)



    print('trial_input_0',trial_input_0.keys())
    start_projection=0
    end_projection = trial_input_0['epochs']['response'][1]
    print('end_projection ',end_projection )


    start_time_0, _ = trial_input_0['epochs']['stim']
    print('start_time_0',start_time_0.shape,start_time_0)

    end_time_0 = np.zeros_like(start_time_0) + end_projection * np.ones_like(start_time_0)
    start_time_0 = np.zeros_like(start_time_0) + start_projection * np.ones_like(start_time_0)


    firing_rate_cue_0 = run_result_0.firing_rate_md.detach().cpu().numpy()
    print("=== firing_rate_cue_0", firing_rate_cue_0.shape)
    firing_rate_list_0 = list(firing_rate_cue_0[start_time_0[i]:end_time_0[i], i, :] for i in range(0, batch_size))
    print('**firing_rate_list_0',firing_rate_list_0[0].shape)
    concate_firing_rate_0 = np.concatenate(firing_rate_list_0, axis=0)
    print("concate_firing_rate_0", concate_firing_rate_0.shape)



    start_time_1=start_time_0#, _ = trial_input_1.epochs['interval']
    end_time_1 = end_time_0#trial_input_1.epochs['interval']
    firing_rate_cue_1 = run_result_1.firing_rate_md.detach().cpu().numpy()
    firing_rate_list_1 = list(firing_rate_cue_1[start_time_1[i]:end_time_1[i], i, :] for i in range(0, batch_size))
    concate_firing_rate_1 = np.concatenate(firing_rate_list_1, axis=0)
    print("concate_firing_rate_1", concate_firing_rate_1.shape)


    concate_firing_rate = np.concatenate((concate_firing_rate_0, concate_firing_rate_1),axis=0)
    print("*** concate_firing_rate", concate_firing_rate.shape)

    pca = PCA(n_components=3)
    pca.fit(concate_firing_rate)

    explained_variance_ratio = pca.explained_variance_ratio_
    print('explained_variance_ratio', explained_variance_ratio)

    concate_firing_rate_transform = pca.transform(concate_firing_rate)

    start_time = np.concatenate((start_time_0, start_time_1), axis=0)
    end_time = np.concatenate((end_time_0, end_time_1), axis=0)
    print('start_time',start_time.shape,start_time)

    time_size = end_time - start_time
    delim = np.cumsum(time_size)
    print('delim', delim)

    print("##concate_firing_rate_transform", concate_firing_rate_transform.shape)
    concate_transform_split = np.split(concate_firing_rate_transform, delim[:-1], axis=0)
    print('concate_transform_split', len(concate_transform_split))

    fig = plt.figure(figsize=(3.0, 3))
    ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])
    colors_1 = sns.color_palette("husl", 8)
    fs=5
    for i in range(0, len(concate_transform_split)):

        if i < batch_size:
            ax.plot(concate_transform_split[i][:, 0], concate_transform_split[i][:, 1], linewidth=0.5,
                    color=colors_1[0], zorder=0)
            ax.scatter(concate_transform_split[i][-1, 0], concate_transform_split[i][-1, 1], linewidth=0.2, marker='o',
                       color='red')


        else:
            ax.plot(concate_transform_split[i][:, 0], concate_transform_split[i][:, 1], linewidth=0.5,
                    color=colors_1[5], zorder=1)
            ax.scatter(concate_transform_split[i][-1, 0], concate_transform_split[i][-1, 1], linewidth=0.2, marker='o',
                       color='blue')

        ax.scatter(concate_transform_split[i][0, 0], concate_transform_split[i][0, 1], linewidth=0.2, marker='*',
                   color='green')
        ax.scatter(concate_transform_split[i][0, 0], concate_transform_split[i][0, 1], linewidth=0.2, marker='*',
                   color='green')

    ax.set_xlabel('integ-PC1', fontsize=fs)
    ax.set_ylabel('integ-PC2', fontsize=fs)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)

    ax.grid(True)
    plt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    plt.title(model_name + '/' + str(idx) + ':' + str(start_time[0]) + '_' + str(end_time[0]), fontsize=5)
    plt.show()
    # plt.savefig(figure_path + 'PCA_conA2B_fail' + str(start_time[0]) + '_' + str(end_time[0]) + '.eps', format='eps',
    #             dpi=1000)



def PCA_h_3D(figure_path,model_name,model_dir,hp):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """

    batch_size=hp['batch_size']
    runnerObj = run.Runner(rule_name=hp['rule_name'], hp=hp, model_dir=model_dir, is_cuda=False, noise_on=True, mode='random_validate')

    trial_input, run_result = runnerObj.run(batch_size=batch_size)
    Vt_list = np.array([Vt.detach().cpu().numpy() for Vt in run_result.V_t])  # .detach().cpu().numpy()
    trial_info = trial_input

    print('trial_info', trial_info.keys())

    target_choice = trial_info['target_choice']
    epochs = trial_info['epochs']
    delay_on = epochs['stim'][1][0]
    response_on = epochs['response'][0][0]


    start_time = 0
    end_time = Vt_list.shape[0]

    start_projection = start_time
    end_projection = end_time


    choice_left = np.where(target_choice > 0)[0]
    choice_right = np.where(target_choice == 0)[0]
    # print('choice_left',choice_left)
    # print('choice_right', choice_right)
    start_time_0 = len(choice_left) * [start_projection]  # np.ones(len(choice_left))
    end_time_0 = len(choice_left) * [end_projection]

    start_time_1 = len(choice_right) * [start_projection]  # np.ones(len(choice_left))
    end_time_1 = len(choice_right) * [end_projection]

    # print('******* start_time_0=',start_time_0)
    # print('******* end_time_0=', end_time_0)

    activity_split_left = Vt_list[:, choice_left, :]
    activity_split_right = Vt_list[:, choice_right, :]

    # print('choice_left',len(choice_left))
    # print('choice_right', len(choice_right))

    firing_rate_list_0 = list(activity_split_left[:, i, :] for i in range(0, len(choice_left)))
    firing_rate_list_1 = list(activity_split_right[:, i, :] for i in range(0, len(choice_right)))
    # print('**firing_rate_list_0', firing_rate_list_0[0].shape)
    concate_firing_rate_0 = np.concatenate(firing_rate_list_0, axis=0)
    concate_firing_rate_1 = np.concatenate(firing_rate_list_1, axis=0)

    concate_firing_rate = np.concatenate((concate_firing_rate_0, concate_firing_rate_1), axis=0)

    pca = PCA(n_components=3)
    pca.fit(concate_firing_rate)

    concate_firing_rate_transform = pca.transform(concate_firing_rate)
    start_time = np.concatenate((start_time_0, start_time_1), axis=0)
    end_time = np.concatenate((end_time_0, end_time_1), axis=0)
    time_size = end_time - start_time
    delim = np.cumsum(time_size)
    # print('delim',delim)
    # print("##concate_firing_rate_transform", concate_firing_rate_transform.shape)

    concate_transform_split = np.split(concate_firing_rate_transform, delim[:], axis=0)

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')  #

    colors_1 = sns.color_palette("husl", 8)
    fs = 5

    print(len(concate_transform_split))
    from matplotlib.collections import LineCollection
    for i in range(0, len(concate_transform_split) - 1):

        traj_idx = concate_transform_split[i]

        if i < len(choice_left):
            ax.plot(traj_idx[:, 0], traj_idx[:, 1], traj_idx[:, 2], linewidth=0.5, color=colors_1[0], zorder=0)
            ax.scatter(traj_idx[-1, 0], traj_idx[-1, 1], traj_idx[-1, 2], linewidth=0.5, marker='o', color='red')
            ax.scatter(traj_idx[delay_on, 0], traj_idx[delay_on, 1], traj_idx[delay_on, 2], marker='o', s=0.1,
                       color='red')
            ax.scatter(traj_idx[response_on, 0], traj_idx[response_on, 1], traj_idx[response_on, 2], marker='o', s=0.1,
                       color='red')
        else:
            ax.plot(traj_idx[:, 0], traj_idx[:, 1], traj_idx[:, 2], linewidth=0.5, color=colors_1[5], zorder=1)
            ax.scatter(traj_idx[-1, 0], traj_idx[-1, 1], traj_idx[-1, 2], linewidth=0.5, marker='o', color='blue')
            ax.scatter(traj_idx[delay_on, 0], traj_idx[delay_on, 1], traj_idx[delay_on, 2], marker='o', s=0.1,
                       color='blue')
            ax.scatter(traj_idx[response_on, 0], traj_idx[response_on, 1], traj_idx[response_on, 2], marker='o', s=0.1,
                       color='blue')

        ax.scatter(traj_idx[0, 0], traj_idx[0, 1], traj_idx[0, 2], linewidth=1, marker='*', color='green')
        ax.scatter(traj_idx[0, 0], traj_idx[0, 1], traj_idx[0, 2], linewidth=1, marker='*', color='green')

    ax.set_xlabel('integ-PC1', fontsize=fs)
    ax.set_ylabel('integ-PC2', fontsize=fs)
    ax.set_zlabel('integ-PC3', fontsize=fs)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    plt.savefig(model_dir + '.png')
    plt.show()


def PCA_h_2D_mean(figure_path,model_name,model_dir,hp):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """

    batch_size=hp['batch_size']
    runnerObj = run.Runner(rule_name=hp['rule_name'], hp=hp, model_dir=model_dir, is_cuda=False, noise_on=True, mode='random_validate')

    trial_input, run_result = runnerObj.run(batch_size=batch_size)
    Vt_list = np.array([Vt.detach().cpu().numpy() for Vt in run_result.V_t])  # .detach().cpu().numpy()
    trial_info = trial_input

    print('trial_info', trial_info.keys())

    target_choice = trial_info['target_choice']
    epochs = trial_info['epochs']
    delay_on = epochs['stim'][1][0]
    response_on = epochs['response'][0][0]


    start_time = 0
    end_time = Vt_list.shape[0]

    start_projection = start_time
    end_projection = end_time


    choice_left = np.where(target_choice > 0)[0]
    choice_right = np.where(target_choice == 0)[0]
    # print('choice_left',choice_left)
    # print('choice_right', choice_right)
    start_time_0 = len(choice_left) * [start_projection]  # np.ones(len(choice_left))
    end_time_0 = len(choice_left) * [end_projection]

    start_time_1 = len(choice_right) * [start_projection]  # np.ones(len(choice_left))
    end_time_1 = len(choice_right) * [end_projection]

    # print('******* start_time_0=',start_time_0)
    # print('******* end_time_0=', end_time_0)

    activity_split_left = Vt_list[:, choice_left, :]
    activity_split_right = Vt_list[:, choice_right, :]

    # print('choice_left',len(choice_left))
    # print('choice_right', len(choice_right))

    firing_rate_list_0 = list(activity_split_left[:, i, :] for i in range(0, len(choice_left)))
    firing_rate_list_1 = list(activity_split_right[:, i, :] for i in range(0, len(choice_right)))
    # print('**firing_rate_list_0', firing_rate_list_0[0].shape)
    concate_firing_rate_0 = np.concatenate(firing_rate_list_0, axis=0)
    concate_firing_rate_1 = np.concatenate(firing_rate_list_1, axis=0)

    concate_firing_rate = np.concatenate((concate_firing_rate_0, concate_firing_rate_1), axis=0)

    pca = PCA(n_components=2)
    pca.fit(concate_firing_rate)

    concate_firing_rate_transform = pca.transform(concate_firing_rate)
    start_time = np.concatenate((start_time_0, start_time_1), axis=0)
    end_time = np.concatenate((end_time_0, end_time_1), axis=0)
    time_size = end_time - start_time
    delim = np.cumsum(time_size)
    # print('delim',delim)
    # print("##concate_firing_rate_transform", concate_firing_rate_transform.shape)

    concate_transform_split = np.split(concate_firing_rate_transform, delim[:], axis=0)

    print(len(concate_transform_split))

    traj_idx_left = np.array([concate_transform_split[i] for i in range(0, len(choice_left) - 1)])

    traj_idx_right = np.array([concate_transform_split[i] for i in range(len(choice_left)+1, len(concate_transform_split) - 1)])


    print('traj_idx_left',traj_idx_left.shape)

    traj_idx_left_mean = np.mean(traj_idx_left,axis=0)
    traj_idx_right_mean = np.mean(traj_idx_right, axis=0)
    print(traj_idx_left_mean.shape)

    fig = plt.figure(figsize=(3.0, 3))
    ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])
    colors_1 = sns.color_palette("husl", 8)
    fs = 5

    ax.plot(traj_idx_left_mean[:, 0],traj_idx_left_mean[:, 1],  linewidth=0.5, color=colors_1[0], zorder=0)
    ax.plot(traj_idx_right_mean[:, 0],traj_idx_right_mean[:, 1],  linewidth=0.5, color=colors_1[5], zorder=0)

    ax.scatter(traj_idx_left_mean[-1, 0],traj_idx_left_mean[-1, 1], linewidth=0.5, marker='o', color='red')
    ax.scatter(traj_idx_left_mean[delay_on, 0], traj_idx_left_mean[delay_on, 1], marker='o', s=0.1,color='red')
    ax.scatter(traj_idx_left_mean[response_on, 0], traj_idx_left_mean[response_on, 1],marker='o', s=0.1,color='red')


    ax.scatter(traj_idx_right_mean[-1, 0],traj_idx_right_mean[-1, 1], linewidth=0.5, marker='o', color='blue')
    ax.scatter(traj_idx_right_mean[delay_on, 0], traj_idx_right_mean[delay_on, 1], marker='o', s=0.1,color='blue')
    ax.scatter(traj_idx_right_mean[response_on, 0], traj_idx_right_mean[response_on, 1],marker='o', s=0.1,color='blue')


    ax.scatter(traj_idx_left_mean[0, 0], traj_idx_left_mean[0, 1], linewidth=1, marker='*', color='green')
    ax.scatter(traj_idx_right_mean[0, 0], traj_idx_right_mean[0, 1], linewidth=1, marker='*', color='green')

    ax.set_xlabel('integ-PC1', fontsize=fs)
    ax.set_ylabel('integ-PC2', fontsize=fs)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    plt.savefig(model_dir + model_name+'_mean.png')
    plt.show()



def plot_activity_h(figure_path,model_name,model_dir,hp):

    fig_path = os.path.join(figure_path, 'plot_activity_h/')
    tools.mkdir_p(fig_path)


    rule_name = hp['rule_name']

    n_rnn = hp['n_rnn']
    n_md = hp['n_md']


    batch_size=hp['batch_size']
    #c = 0.005*np.random.choice([-6.4, -12.8, -25.6, -51.2, 6.4, 12.8, 25.6, 51.2])


    pfc_mean_0, h_mean_0, h_mean_0,epochs = get_neurons_activity_mode_test1(context_name=hp['rule_name'],hp=hp,type=hp['type'],
                                                model_dir=model_dir,gamma_bar=0.5,choice=0,batch_size=batch_size)
    print('pfc_mean_0',pfc_mean_0)
    delay_on = epochs['stim'][1][0]
    response_on = epochs['response'][0][0]
    response_off = epochs['response'][1][0]

    ########## exc ###################
    #'''

    fig = plt.figure(figsize=(4.0, 3))
    ax = fig.add_axes([0.15, 0.15, 0.75, 0.7])

    for i in range(128):
        ax.plot(h_mean_0[:,i],label=str(i))
    ax.axvspan(delay_on - 1, response_on - 1, color='grey', alpha=0.2, label='delay_on')

    # plt.xlim([0,35])
    # plt.ylim([0, 10])


    file_name = hp['type']+'_model'+str(hp['model_idx'])
    plt.title(file_name+'; delay='+str(hp['stim_delay']),fontsize=10)

    plt.savefig(fig_path+file_name+'_delay'+str(hp['stim_delay'])+'h.png')
    plt.show()

def PCA_h_2D(figure_path,model_name,model_dir,hp):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    batch_size = hp['batch_size']
    rng = hp['rng']

    c = 0.005 * rng.choice([-25.6, -51.2, 25.6, 51.2], (batch_size,))

    batch_size=hp['batch_size']
    runnerObj = run.Runner(rule_name=hp['rule_name'], hp=hp, model_dir=model_dir,
                           is_cuda=False, noise_on=True, mode='pca')

    trial_input, run_result = runnerObj.run(batch_size=batch_size,c=c)
    Vt_list = np.array([Vt.detach().cpu().numpy() for Vt in run_result.V_t])  # .detach().cpu().numpy()
    trial_info = trial_input

    print('trial_info', trial_info.keys())

    target_choice = trial_info['target_choice']
    epochs = trial_info['epochs']
    delay_on = epochs['stim'][1][0]
    response_on = epochs['response'][0][0]
    response_off = epochs['response'][1][0]


    start_time = 0
    end_time = response_off#Vt_list.shape[0]

    start_projection = start_time
    end_projection = end_time


    choice_left = np.where(target_choice > 0)[0]
    choice_right = np.where(target_choice == 0)[0]
    # print('choice_left',choice_left)
    # print('choice_right', choice_right)
    start_time_0 = len(choice_left) * [start_projection]  # np.ones(len(choice_left))
    end_time_0 = len(choice_left) * [end_projection]

    start_time_1 = len(choice_right) * [start_projection]  # np.ones(len(choice_left))
    end_time_1 = len(choice_right) * [end_projection]

    # print('******* start_time_0=',start_time_0)
    # print('******* end_time_0=', end_time_0)

    activity_split_left = Vt_list[:, choice_left, :]
    activity_split_right = Vt_list[:, choice_right, :]

    # print('choice_left',len(choice_left))
    # print('choice_right', len(choice_right))

    firing_rate_list_0 = list(activity_split_left[start_projection:end_projection, i, :] for i in range(0, len(choice_left)))
    firing_rate_list_1 = list(activity_split_right[start_projection:end_projection, i, :] for i in range(0, len(choice_right)))
    # print('**firing_rate_list_0', firing_rate_list_0[0].shape)
    concate_firing_rate_0 = np.concatenate(firing_rate_list_0, axis=0)
    concate_firing_rate_1 = np.concatenate(firing_rate_list_1, axis=0)

    concate_firing_rate = np.concatenate((concate_firing_rate_0, concate_firing_rate_1), axis=0)

    pca = PCA(n_components=2)
    pca.fit(concate_firing_rate)

    concate_firing_rate_transform = pca.transform(concate_firing_rate)
    start_time = np.concatenate((start_time_0, start_time_1), axis=0)
    end_time = np.concatenate((end_time_0, end_time_1), axis=0)
    time_size = end_time - start_time
    delim = np.cumsum(time_size)
    # print('delim',delim)
    # print("##concate_firing_rate_transform", concate_firing_rate_transform.shape)

    concate_transform_split = np.split(concate_firing_rate_transform, delim[:], axis=0)

    fig = plt.figure(figsize=(3.0, 3.1))
    ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])
    colors_1 = sns.color_palette("husl", 8)
    fs = 8

    print(len(concate_transform_split))

    for i in range(0, len(concate_transform_split) - 1):

        traj_idx = concate_transform_split[i]


        points = traj_idx.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Normalize time for color mapping
        norm = plt.Normalize(0, traj_idx.shape[0] - 1)
        cmap = cm.viridis  # You can try others like 'plasma', 'cool', etc.
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(np.arange(traj_idx.shape[0]))
        lc.set_linewidth(0.7)
        ax.add_collection(lc)

        if i < len(choice_left):
            ax.plot(traj_idx[:, 0], traj_idx[:, 1], linewidth=0.5, color=colors_1[0], zorder=0)
            ax.scatter(traj_idx[-1, 0], traj_idx[-1, 1], linewidth=0.5, marker='o', color='red', zorder=10)
            #ax.scatter(traj_idx[delay_on, 0], traj_idx[delay_on, 1], marker='o', s=2,color='red')
            #ax.scatter(traj_idx[response_on, 0], traj_idx[response_on, 1], marker='*', s=2,color='red')
        else:
            ax.plot(traj_idx[:, 0], traj_idx[:, 1], linewidth=0.5, color=colors_1[5], zorder=1)
            ax.scatter(traj_idx[-1, 0], traj_idx[-1, 1], linewidth=0.5, marker='o', color='blue', zorder=10)
            #ax.scatter(traj_idx[delay_on, 0], traj_idx[delay_on, 1], marker='o', s=2,color='blue')
            #ax.scatter(traj_idx[response_on, 0], traj_idx[response_on, 1],marker='*', s=2,color='blue')

        ax.scatter(traj_idx[0, 0], traj_idx[0, 1], linewidth=1, marker='*', color='orange', zorder=10)
        ax.scatter(traj_idx[0, 0], traj_idx[0, 1], linewidth=1, marker='*', color='orange', zorder=10)

    ax.set_xlabel('integ-PC1', fontsize=fs)
    ax.set_ylabel('integ-PC2', fontsize=fs)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)

    file_name = hp['type']+'_model_'+str(hp['model_idx'])
    plt.title(file_name+'h; delay='+str(hp['stim_delay']),fontsize=8)


    plt.savefig(figure_path+file_name+'_delay'+str(hp['stim_delay'])+ '_h.pdf')

    plt.show()

def PCA_h_2D_velocity(figure_path,model_name,model_dir,hp):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """

    batch_size=hp['batch_size']
    runnerObj = run.Runner(rule_name=hp['rule_name'], hp=hp, model_dir=model_dir, is_cuda=False, noise_on=True, mode='random_validate')

    trial_input, run_result = runnerObj.run(batch_size=batch_size)
    Vt_list = np.array([Vt.detach().cpu().numpy() for Vt in run_result.V_t])  # .detach().cpu().numpy()
    trial_info = trial_input

    print('trial_info', trial_info.keys())

    target_choice = trial_info['target_choice']
    epochs = trial_info['epochs']
    delay_on = epochs['stim'][1][0]
    response_on = epochs['response'][0][0]
    response_off = epochs['response'][1][0]

    start_time = 0
    end_time = response_off-1  # Vt_list.shape[0]
    print('end_time',end_time)

    start_projection = start_time
    end_projection = end_time


    choice_left = np.where(target_choice > 0)[0]
    choice_right = np.where(target_choice == 0)[0]
    # print('choice_left',choice_left)
    # print('choice_right', choice_right)
    start_time_0 = len(choice_left) * [start_projection]  # np.ones(len(choice_left))
    end_time_0 = len(choice_left) * [end_projection]

    start_time_1 = len(choice_right) * [start_projection]  # np.ones(len(choice_left))
    end_time_1 = len(choice_right) * [end_projection]

    # print('******* start_time_0=',start_time_0)
    # print('******* end_time_0=', end_time_0)

    activity_split_left = Vt_list[:, choice_left, :]
    activity_split_right = Vt_list[:, choice_right, :]

    # print('choice_left',len(choice_left))
    # print('choice_right', len(choice_right))

    firing_rate_list_0 = list(activity_split_left[start_projection:end_projection, i, :] for i in range(0, len(choice_left)))
    firing_rate_list_1 = list(activity_split_right[start_projection:end_projection, i, :] for i in range(0, len(choice_right)))
    # print('**firing_rate_list_0', firing_rate_list_0[0].shape)
    concate_firing_rate_0 = np.concatenate(firing_rate_list_0, axis=0)
    concate_firing_rate_1 = np.concatenate(firing_rate_list_1, axis=0)

    concate_firing_rate = np.concatenate((concate_firing_rate_0, concate_firing_rate_1), axis=0)

    pca = PCA(n_components=2)
    pca.fit(concate_firing_rate)

    concate_firing_rate_transform = pca.transform(concate_firing_rate)
    start_time = np.concatenate((start_time_0, start_time_1), axis=0)
    end_time = np.concatenate((end_time_0, end_time_1), axis=0)
    time_size = end_time - start_time
    delim = np.cumsum(time_size)
    # print('delim',delim)
    # print("##concate_firing_rate_transform", concate_firing_rate_transform.shape)

    concate_transform_split = np.split(concate_firing_rate_transform, delim[:], axis=0)


    print(len(concate_transform_split))

    traj_idx_left = np.array([concate_transform_split[i] for i in range(0, len(choice_left) - 1)])
    traj_idx_right = np.array([concate_transform_split[i] for i in range(len(choice_left)+1, len(concate_transform_split) - 1)])

    traj_idx = np.array([concate_transform_split[i] for i in range(len(concate_transform_split) - 1)])
    use_speed='both'
    if use_speed=='left':
        traj_selected = traj_idx_left
    if use_speed=='right':
        traj_selected = traj_idx_right
    if use_speed=='both':
        traj_selected = traj_idx


    v_list = []
    for n_trial in range(traj_selected.shape[0]-1):
        x = traj_selected[n_trial, :, 0]
        y = traj_selected[n_trial, :, 0]

        vx = [x[i + 1] - x[i] for i in range(x.shape[0] - 1)]
        vy = [y[i + 1] - y[i] for i in range(y.shape[0] - 1)]

        v = [np.sqrt(vx[i] ** 2 + vy[i] ** 2) for i in range(x.shape[0] - 1)]

        v_list.append(v)

    v_list = np.array(v_list)
    print(v_list.shape)

    v = np.mean(v_list,axis=0)

    delay_time = response_on-delay_on

    fig = plt.figure(figsize=(2.6, 2))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.6])
    fs=10
    plt.plot(v,linewidth=2,c='gray')
    ax.axvspan(delay_on-1, response_on-1, color='grey', alpha=0.15,label='delay_on')
    #ax.axvspan(response_on, response_on+0.5, color='grey', label='delay_off')


    ax.set_xlabel('time', fontsize=fs)
    ax.set_ylabel('velocity', fontsize=fs)

    plt.ylim([0,0.3])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.legend()

    ticks = [0, 20, 40]

    if hp['stim_delay']==300:
        ticks = [0, 20, 40]
    elif hp['stim_delay']==900:
        ticks = [0, 25, 50, 75]

    elif hp['stim_delay'] == 1800:
        ticks = [0,50,100]

    # ticks = [0, 50, 100]
    plt.xticks(ticks, [int(t*20) for t in ticks])  # Show [0, 5, 10] instead of [0, 50, 100]



    #plt.title(model_name+'\n delay='+str(hp['stim_delay'])+';'+use_speed,fontsize=5)
    file_name = hp['type']+'_model_'+str(hp['model_idx'])
    plt.title(file_name+'; delay='+str(hp['stim_delay'])+'\n'+use_speed,fontsize=8)


    # plt.savefig(figure_path+file_name+ '_vel.png')
    plt.savefig(figure_path+file_name+'_delay'+str(hp['stim_delay'])+ '_h_velocity.pdf')

    plt.show()





def PCA_h_2D_velocity_mean(figure_path,model_name,model_dir,hp):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    file_name = hp['type']+'_model'+str(hp['model_idx'])+'_delay'+str(hp['stim_delay'])
    batch_size=hp['batch_size']
    runnerObj = run.Runner(rule_name=hp['rule_name'], hp=hp, model_dir=model_dir, is_cuda=False, noise_on=True, mode='random_validate')

    trial_input, run_result = runnerObj.run(batch_size=batch_size)
    Vt_list = np.array([Vt.detach().cpu().numpy() for Vt in run_result.V_t])  # .detach().cpu().numpy()
    trial_info = trial_input

    print('trial_info', trial_info.keys())

    target_choice = trial_info['target_choice']
    epochs = trial_info['epochs']
    delay_on = epochs['stim'][1][0]
    response_on = epochs['response'][0][0]
    response_off = epochs['response'][1][0]

    start_time = 0
    end_time = response_off  # Vt_list.shape[0]

    start_projection = start_time
    end_projection = end_time


    choice_left = np.where(target_choice > 0)[0]
    choice_right = np.where(target_choice == 0)[0]
    # print('choice_left',choice_left)
    # print('choice_right', choice_right)
    start_time_0 = len(choice_left) * [start_projection]  # np.ones(len(choice_left))
    end_time_0 = len(choice_left) * [end_projection]

    start_time_1 = len(choice_right) * [start_projection]  # np.ones(len(choice_left))
    end_time_1 = len(choice_right) * [end_projection]

    # print('******* start_time_0=',start_time_0)
    # print('******* end_time_0=', end_time_0)

    activity_split_left = Vt_list[:, choice_left, :]
    activity_split_right = Vt_list[:, choice_right, :]

    # print('choice_left',len(choice_left))
    # print('choice_right', len(choice_right))

    firing_rate_list_0 = list(activity_split_left[start_projection:end_projection, i, :] for i in range(0, len(choice_left)))
    firing_rate_list_1 = list(activity_split_right[start_projection:end_projection, i, :] for i in range(0, len(choice_right)))
    # print('**firing_rate_list_0', firing_rate_list_0[0].shape)
    concate_firing_rate_0 = np.concatenate(firing_rate_list_0, axis=0)
    concate_firing_rate_1 = np.concatenate(firing_rate_list_1, axis=0)

    concate_firing_rate = np.concatenate((concate_firing_rate_0, concate_firing_rate_1), axis=0)

    pca = PCA(n_components=2)
    pca.fit(concate_firing_rate)

    concate_firing_rate_transform = pca.transform(concate_firing_rate)
    start_time = np.concatenate((start_time_0, start_time_1), axis=0)
    end_time = np.concatenate((end_time_0, end_time_1), axis=0)
    time_size = end_time - start_time
    delim = np.cumsum(time_size)
    # print('delim',delim)
    # print("##concate_firing_rate_transform", concate_firing_rate_transform.shape)

    concate_transform_split = np.split(concate_firing_rate_transform, delim[:], axis=0)


    print(len(concate_transform_split))

    traj_idx_left = np.array([concate_transform_split[i] for i in range(0, len(choice_left) - 1)])
    traj_idx_right = np.array([concate_transform_split[i] for i in range(len(choice_left)+1, len(concate_transform_split) - 1)])

    traj_idx_left_mean = np.mean(traj_idx_left, axis=0)
    traj_idx_right_mean = np.mean(traj_idx_right, axis=0)


    print('traj_idx_left',traj_idx_left.shape)



    x = traj_idx_right_mean[:, 0]
    y = traj_idx_right_mean[:, 0]

    vx = [x[i + 1] - x[i] for i in range(x.shape[0] - 1)]
    vy = [y[i + 1] - y[i] for i in range(y.shape[0] - 1)]

    v = [np.sqrt(vx[i] ** 2 + vy[i] ** 2) for i in range(x.shape[0] - 1)]


    fig = plt.figure(figsize=(5.0, 3))
    ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])
    fs=10
    plt.plot(v)
    ax.axvspan(delay_on, delay_on, color='grey', label='cue_on')
    ax.axvspan(response_on, response_on, color='grey', label='cue_on')


    ax.set_xlabel('time', fontsize=fs)
    ax.set_ylabel('velocity', fontsize=fs)

    plt.ylim([0,0.6])


    plt.title(model_name+'\n delay='+str(hp['stim_delay']),fontsize=8)
    plt.savefig(figure_path+file_name+'_'+str(hp['stim_delay'])+ '_vel.png')

    plt.show()


def PCA_pfc_2D(figure_path,model_name,model_dir,hp):
    """
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    """
    fig_path = os.path.join(figure_path, 'PCA_pfc_2D/')
    tools.mkdir_p(fig_path)



    batch_size=hp['batch_size']
    runnerObj = run.Runner(rule_name=hp['rule_name'], hp=hp, model_dir=model_dir, is_cuda=False, noise_on=True, mode='test1')

    trial_input, run_result = runnerObj.run(batch_size=batch_size)
    Vt_list = np.array([Vt.detach().cpu().numpy() for Vt in run_result.firing_rate_binder])  # .detach().cpu().numpy()
    trial_info = trial_input

    print('trial_info', trial_info.keys())

    target_choice = trial_info['target_choice']
    epochs = trial_info['epochs']
    delay_on = epochs['stim'][1][0]
    response_on = epochs['response'][0][0]
    response_off = epochs['response'][1][0]


    start_time = 0
    end_time = response_off#Vt_list.shape[0]

    start_projection = start_time
    end_projection = end_time


    choice_left = np.where(target_choice > 0)[0]
    choice_right = np.where(target_choice == 0)[0]
    # print('choice_left',choice_left)
    # print('choice_right', choice_right)
    start_time_0 = len(choice_left) * [start_projection]  # np.ones(len(choice_left))
    end_time_0 = len(choice_left) * [end_projection]

    start_time_1 = len(choice_right) * [start_projection]  # np.ones(len(choice_left))
    end_time_1 = len(choice_right) * [end_projection]

    # print('******* start_time_0=',start_time_0)
    # print('******* end_time_0=', end_time_0)

    activity_split_left = Vt_list[:, choice_left, :]
    activity_split_right = Vt_list[:, choice_right, :]

    # print('choice_left',len(choice_left))
    # print('choice_right', len(choice_right))

    firing_rate_list_0 = list(activity_split_left[start_projection:end_projection, i, :] for i in range(0, len(choice_left)))
    firing_rate_list_1 = list(activity_split_right[start_projection:end_projection, i, :] for i in range(0, len(choice_right)))
    # print('**firing_rate_list_0', firing_rate_list_0[0].shape)
    concate_firing_rate_0 = np.concatenate(firing_rate_list_0, axis=0)
    concate_firing_rate_1 = np.concatenate(firing_rate_list_1, axis=0)

    concate_firing_rate = np.concatenate((concate_firing_rate_0, concate_firing_rate_1), axis=0)

    pca = PCA(n_components=2)
    pca.fit(concate_firing_rate)

    concate_firing_rate_transform = pca.transform(concate_firing_rate)
    start_time = np.concatenate((start_time_0, start_time_1), axis=0)
    end_time = np.concatenate((end_time_0, end_time_1), axis=0)
    time_size = end_time - start_time
    delim = np.cumsum(time_size)
    # print('delim',delim)
    # print("##concate_firing_rate_transform", concate_firing_rate_transform.shape)

    concate_transform_split = np.split(concate_firing_rate_transform, delim[:], axis=0)

    fig = plt.figure(figsize=(3.0, 3.1))
    ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])
    colors_1 = sns.color_palette("husl", 8)
    fs = 8

    print(len(concate_transform_split))

    for i in range(0, len(concate_transform_split) - 1):

        traj_idx = concate_transform_split[i]


        points = traj_idx.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Normalize time for color mapping
        norm = plt.Normalize(0, traj_idx.shape[0] - 1)
        cmap = cm.viridis  # You can try others like 'plasma', 'cool', etc.
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(np.arange(traj_idx.shape[0]))
        lc.set_linewidth(0.7)
        ax.add_collection(lc)

        if i < len(choice_left):
            ax.plot(traj_idx[:, 0], traj_idx[:, 1], linewidth=0.5, color=colors_1[0], zorder=0)
            ax.scatter(traj_idx[-1, 0], traj_idx[-1, 1], linewidth=0.5, marker='o', color='red', zorder=10)
            #ax.scatter(traj_idx[delay_on, 0], traj_idx[delay_on, 1], marker='o', s=2,color='red')
            #ax.scatter(traj_idx[response_on, 0], traj_idx[response_on, 1], marker='*', s=2,color='red')
        else:
            ax.plot(traj_idx[:, 0], traj_idx[:, 1], linewidth=0.5, color=colors_1[5], zorder=1)
            ax.scatter(traj_idx[-1, 0], traj_idx[-1, 1], linewidth=0.5, marker='o', color='blue', zorder=10)
            #ax.scatter(traj_idx[delay_on, 0], traj_idx[delay_on, 1], marker='o', s=2,color='blue')
            #ax.scatter(traj_idx[response_on, 0], traj_idx[response_on, 1],marker='*', s=2,color='blue')

        ax.scatter(traj_idx[0, 0], traj_idx[0, 1], linewidth=1, marker='*', color='orange', zorder=10)
        ax.scatter(traj_idx[0, 0], traj_idx[0, 1], linewidth=1, marker='*', color='orange', zorder=10)

    ax.set_xlabel('integ-PC1', fontsize=fs)
    ax.set_ylabel('integ-PC2', fontsize=fs)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    plt.title(model_name+'\n delay='+str(hp['stim_delay']),fontsize=5)

    file_name = hp['type']+'_model'+str(hp['model_idx'])+'_delay'+str(hp['stim_delay'])

    plt.savefig(fig_path+file_name+ '.png')
    plt.savefig(fig_path+file_name+ '.pdf')

    plt.show()
























