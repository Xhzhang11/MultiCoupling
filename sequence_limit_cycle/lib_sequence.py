

from __future__ import division
import os,sys
import seaborn as sns
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import run
import tools


def get_epoch(hp):
    # print('************ perform HL_task')
    dt = hp['dt']
    batch_size = 1
    rng = hp['rng']


    fix_on = (rng.uniform(0, 0, batch_size) / dt).astype(int)
    stim1_on = (rng.uniform(0, 0, batch_size) / dt).astype(int)
    stim_duration = int(hp['stim_duration'] / dt)
    stim1_off = stim1_on + stim_duration

    stim_delay = int(hp['stim_delay'] / dt)
    response_on = stim1_off + stim_delay
    response_duration = int(hp['response_time'] / dt)

    response_off = response_on + response_duration


    epoch = {'fix': (fix_on, stim1_on),
                    'stim': (stim1_on, stim1_off),
                    'response': (response_on, response_off)}

    print(epoch)

    return epoch


def generate_test_trial(context_name,hp,model_dir,
                        c,
                        gamma_bar,
                        choice,
                        batch_size=1):

    rng  = hp['rng']

    runnerObj = run.Runner(rule_name=context_name, hp=hp,model_dir=model_dir, is_cuda=False, noise_on=True,mode='pca')

    trial_input, run_result = runnerObj.run(batch_size=batch_size,
                                            c=c,
                                            gamma_bar=gamma_bar,
                                            choice=choice)


    return trial_input, run_result

def get_neurons_activity_mode(context_name,hp,type,model_dir,c,gamma_bar,choice,batch_size):
    hp['type']=type

    runnerObj = run.Runner(rule_name=context_name, hp=hp, model_dir=model_dir, is_cuda=False, noise_on=False,mode='pca')
    trial, run_result = runnerObj.run(batch_size=batch_size, c=c,gamma_bar=gamma_bar,choice=choice)
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





def plot_activity_peak_order(figure_path,data_path,model_dir,hp):

    fig_path = os.path.join(figure_path, 'plot_activity_peak_order/')
    tools.mkdir_p(fig_path)


    batch_size = hp['batch_size']
    rng = hp['rng']
    c = 0.005 * rng.choice([-51.2,51.2], (batch_size,))

    # pfc_mean_0, MD_mean_0, h_mean_0,epochs = get_neurons_activity_mode(context_name=hp['rule_name'],hp=hp,type=hp['type'],
    #                                             model_dir=model_dir,c=c,gamma_bar=0.5,choice=0,batch_size=batch_size)


    epochs = get_epoch(hp)
    #np.save(data_path + 'pfc_mean_0.npy', pfc_mean_0)
    pfc_mean_0 = np.load(data_path + 'pfc_mean_0.npy')




    delay_on = epochs['stim'][1][0]
    response_on = epochs['response'][0][0]
    response_off = epochs['response'][1][0]

    start_time = delay_on+1
    end_time = response_on+1#response_off-1#response_on+5


    ################ vis sorted #########################
    func_activity_threshold_0 = 2

    data_0 = pfc_mean_0[start_time:end_time, :]




    print('data_0:',data_0.shape)

    max_firing_rate_vis = np.max(data_0, axis=0)
    pick_idx_vis = np.argwhere(max_firing_rate_vis > func_activity_threshold_0).squeeze()

    data_0 = data_0[:, pick_idx_vis]
    peak_time_vis = np.argmax(data_0, axis=0)

    peak_order_vis = np.argsort(peak_time_vis, axis=0)
    data_0 = data_0[:, peak_order_vis]


    for i in range(0, data_0.shape[1]):
        peak = np.max(data_0[:, i])
        if peak<0:
            data_0[:, i] = 0
        else:
            data_0[:, i] = data_0[:, i] / peak
            #print("np.max(data[:, i])",np.max(data[:, i]))
    #'''
    # Prepare grid for plotting
    time_step=20
    X_0, Y_0 = np.mgrid[0:data_0.shape[0]*time_step:time_step, 0:data_0.shape[1]]

    fig = plt.figure(figsize=(3, 2.5))
    ax = fig.add_axes([0.1, 0.1, 0.9, 0.8])


    fs = 15

    # Make the plot
    #cmap = plt.get_cmap('viridis')#viridis_r
    plt.pcolormesh(X_0, Y_0, data_0)
    m = cm.ScalarMappable(cmap=mpl.rcParams["image.cmap"])#cmap=mpl.rcParams["image.cmap"]
    m.set_array([0, 1])
    m.set_clim(vmin=0, vmax=1)
    cbar = fig.colorbar(m, ax=ax, aspect=15)
    cbar.set_ticks([0,  1])
    cbar.ax.tick_params(labelsize=fs)


    file_name = hp['type'] + '-model' + str(hp['model_idx']) + '-delay' + str(hp['stim_delay'])
    plt.title(file_name,fontsize=8)
    plt.savefig(fig_path + file_name + '_sorting.pdf')
    plt.show()
    #
    # fig = plt.figure(figsize=(4.0, 3))
    # ax = fig.add_axes([0.15, 0.15, 0.75, 0.7])
    # plt.title(file_name, fontsize=8)
    #
    # for i in range(128):
    #     ax.plot(pfc_mean_0[:, i], label=str(i))
    # ax.axvspan(delay_on - 1, response_on - 1, color='grey', alpha=0.2, label='delay_on')
    #
    # # plt.xlim([0,35])
    # # plt.ylim([0, 10])
    #
    # file_name = hp['type'] + '_model' + str(hp['model_idx']) + '_delay' + str(hp['stim_delay'])
    # plt.savefig(fig_path + file_name + '_pfc.pdf')
    # plt.show()

    #

