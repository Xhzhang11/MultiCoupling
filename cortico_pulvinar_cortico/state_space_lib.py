"""
This file contains functions that test the behavior of the model
These functions generally involve some psychometric measurements of the model,
for example performance in decision-making tasks as a function of input strength

These measurements are important as they show whether the network exhibits
some critically important computations, including integration and generalization.
"""


from __future__ import division

import os,sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import numpy as np
from matplotlib import pyplot as plt
import run
import tools
import pdb
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import PCA
c_perf = sns.color_palette("hls", 8)#muted
from scipy.optimize import least_squares
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.stats import sem
fs=5
from scipy.optimize import curve_fit
def get_epoch(model_dir,rule_name,hp):

    runnerObj = run.Runner(rule_name=rule_name, hp=hp,model_dir=model_dir, is_cuda=False, noise_on=True,mode='test')
    trial_input, run_result = runnerObj.run(batch_size=1, gaussian_center1=1, gaussian_center2=0.5,cue_sign=1)
    stim1_on, stim1_off = trial_input.epochs['stim1']
    stim2_on, stim2_off = trial_input.epochs['stim2']
    response_on, response_off = trial_input.epochs['response']


    epoch = {'stim2_on':stim2_on,
             'stim2_off':stim2_off,
             'stim1_on':stim1_on,
             'stim1_off':stim1_off,
             'response_on':response_on,
             'response_off':response_off}
    #print('epoch',epoch)

    return epoch


def generate_test_trial(model_dir,hp,rule_name,batch_size=1,
                        gaussian_center1=0,
                        gaussian_center2=0,
                        cue_sign=None):


    if cue_sign is None:
        cue_sign = hp['rng'].choice([1,-1], (batch_size,))
    else:
        cue_sign = hp['rng'].choice([cue_sign], (batch_size,))


    runnerObj = run.Runner(rule_name=rule_name, hp=hp,model_dir=model_dir, is_cuda=False, noise_on=True,mode='test')

    trial_input, run_result = runnerObj.run(batch_size=batch_size,
                                            gaussian_center1=gaussian_center1,
                                            gaussian_center2=gaussian_center2,
                                            cue_sign=cue_sign)

    return trial_input, run_result


def get_neurons_activity_mean(model_dir,hp,batch_size=1,
                                rule_name='both',
                                gaussian_center1=0,
                                gaussian_center2=0,
                                cue_sign=None):

    if cue_sign is None:
        cue_sign = hp['rng'].choice([1,-1], (batch_size,))
    else:
        cue_sign = hp['rng'].choice([cue_sign], (batch_size,))

    runnerObj = run.Runner(rule_name=rule_name, hp=hp, model_dir=model_dir, is_cuda=False, noise_on=True, mode='test')
    trial_input, run_result = runnerObj.run(batch_size=batch_size,
                                            gaussian_center1=gaussian_center1,
                                            gaussian_center2=gaussian_center2,
                                            cue_sign=cue_sign)

    #### average value over batch_sizes for hidden state
    fr_pfc = run_result.firing_rate_binder_pfc.detach().cpu().numpy()
    fr_pfc_list = list([fr_pfc[:, i, :] for i in range(batch_size)])
    fr_pfc_mean = np.mean(np.array(fr_pfc_list), axis=0)

    fr_parietal = run_result.firing_rate_binder_parietal.detach().cpu().numpy()
    fr_parietal_list = list([fr_parietal[:, i, :] for i in range(batch_size)])
    fr_parietal_mean = np.mean(np.array(fr_parietal_list), axis=0)




    return fr_pfc_mean, fr_parietal_mean




def plot_encode_neuron(fig_path,model_name, model_dir, idx, hp):
    '''
     firing_rate=(times, batch_size, num_units)
     firing_rate_mean = (times, num_units)
    '''

    figure_path = os.path.join(fig_path, 'find_selected_neuron'+'/')
    tools.mkdir_p(figure_path)

    batch_size = 22

    epoch_0 = get_epoch(model_dir=model_dir,rule_name='retro',hp=hp)
    cue_on_0=epoch_0['cue_on'][0];cue_off_0=epoch_0['cue_off'][0]
    stim1_on_0=epoch_0['stim1_on'][0];stim1_off_0=epoch_0['stim1_off'][0];response_on_0=epoch_0['response_on'][0]
    response_off_0 = epoch_0['response_off'][0]
    print('epoch_0', epoch_0)
    epoch_1 = get_epoch(model_dir=model_dir, rule_name='prosp', hp=hp)
    cue_on_1 = epoch_1['cue_on'][0];
    cue_off_1 = epoch_1['cue_off'][0]
    stim1_on_1 = epoch_1['stim1_on'][0];
    stim1_off_1 = epoch_1['stim1_off'][0];
    response_on_1 = epoch_1['response_on'][0]
    response_off_1 = epoch_1['response_off'][0]
    print('epoch_1', epoch_1)
    exc = int(512*0.75)
    start = cue_on_0
    end=cue_off_0+5


    firing_rate_up_ret = get_allangle_activity_mean(model_dir,hp,batch_size=batch_size,rule_name='retro',cue_sign=1)
    firing_rate_down_ret =get_allangle_activity_mean(model_dir, hp, batch_size=batch_size,rule_name='retro',cue_sign=-1)
    firing_rate_up_pro = get_allangle_activity_mean(model_dir, hp, batch_size=batch_size,rule_name='prosp',cue_sign=1)
    firing_rate_down_pro = get_allangle_activity_mean(model_dir, hp, batch_size=batch_size,rule_name='prosp',cue_sign=-1)

    np.save(figure_path + 'firing_rate_up_ret.npy',   firing_rate_up_ret)
    np.save(figure_path + 'firing_rate_down_ret.npy', firing_rate_down_ret)
    np.save(figure_path + 'firing_rate_up_pro.npy',   firing_rate_up_pro)
    np.save(figure_path + 'firing_rate_down_pro.npy', firing_rate_down_pro)

    firing_rate_up_ret   = np.load(figure_path + 'firing_rate_up_ret.npy')
    firing_rate_down_ret = np.load(figure_path + 'firing_rate_down_ret.npy')
    firing_rate_up_pro   = np.load(figure_path + 'firing_rate_up_pro.npy')
    firing_rate_down_pro = np.load(figure_path + 'firing_rate_down_pro.npy')

    diff = np.mean(firing_rate_up_ret[start:end,:], axis=0) - np.mean(firing_rate_down_ret[start:end,:],axis=0)
    diff = diff[0:exc]

    encode_up_cell = np.argwhere(diff > 0.1)[:, 0]
    encode_down_cell = np.argwhere(diff < -0.1)[:, 0]
    print(encode_up_cell.shape, encode_up_cell)
    print(encode_down_cell.shape, encode_down_cell)


    for cell in np.array(encode_up_cell):
        fig, axs = plt.subplots(1, 2, figsize=(7, 3.2))
        fig.subplots_adjust(top=0.85, bottom=0.1, right=0.95, left=0.06, hspace=0.2, wspace=0.3)
        fig.suptitle(model_name + '/' + str(idx) +'; encode up:'+str(cell), fontsize=9)

        axs[0].plot(firing_rate_up_ret[:, cell], color='r', label='up')
        axs[0].plot(firing_rate_down_ret[:, cell], color='g', label='down')

        axs[1].plot(firing_rate_up_pro[:, cell], color='r', label='up')
        axs[1].plot(firing_rate_down_pro[:, cell], color='g', label='down')

        axs[0].axvspan(stim1_on_0 - 1, stim1_off_0 - 1, color='lightgray')
        axs[0].axvspan(cue_on_0 - 1, cue_off_0 - 1, color='lightgray')
        axs[0].axvspan(response_on_0 - 1, response_on_0 + 1, color='lightgray')

        axs[1].axvspan(stim1_on_1 - 1, stim1_off_1 - 1, color='lightgray')
        axs[1].axvspan(cue_on_1 - 1, cue_off_1 - 1, color='lightgray')
        axs[1].axvspan(response_on_1 - 1, response_on_1 + 1, color='lightgray')

        # for i in range(2):
        #     axs[i].set_ylim([0, 3])
        #     #axs[0].legend()

        axs[0].set_title('retro', fontsize=7)
        axs[1].set_title('prosp', fontsize=7)

        plt.savefig(figure_path + 'up_' + str(cell) + '.png')
        #plt.show()


    for cell in np.array(encode_down_cell):
        fig, axs = plt.subplots(1, 2, figsize=(7, 3.2))
        fig.subplots_adjust(top=0.85, bottom=0.1, right=0.95, left=0.06, hspace=0.2, wspace=0.3)
        fig.suptitle(model_name + '/' + str(idx) +'; encode down: '+str(cell), fontsize=9)

        axs[0].plot(firing_rate_up_ret[:, cell], color='r', label='up')
        axs[0].plot(firing_rate_down_ret[:, cell], color='g', label='down')

        axs[1].plot(firing_rate_up_pro[:, cell], color='r', label='up')
        axs[1].plot(firing_rate_down_pro[:, cell], color='g', label='down')

        axs[0].axvspan(stim1_on_0 - 1, stim1_off_0 - 1, color='lightgray')
        axs[0].axvspan(cue_on_0 - 1, cue_off_0 - 1, color='lightgray')
        axs[0].axvspan(response_on_0 - 1, response_on_0 + 1, color='lightgray')

        axs[1].axvspan(stim1_on_1 - 1, stim1_off_1 - 1, color='lightgray')
        axs[1].axvspan(cue_on_1 - 1, cue_off_1 - 1, color='lightgray')
        axs[1].axvspan(response_on_1 - 1, response_on_1 + 1, color='lightgray')

        # for i in range(2):
        #     axs[i].set_ylim([0, 3])
        #     #axs[0].legend()

        axs[0].set_title('retro', fontsize=7)
        axs[1].set_title('prosp', fontsize=7)

        plt.savefig(figure_path + 'down_' + str(cell) + '.png')
        #plt.show()



def generate_for_pca_pfc(model_dir,hp,batch_size=0,start_proj=0,end_proj=0,
                    rule_name='retro',
                    select_cell = None,
                    gaussian_center1=0,
                    gaussian_center2=0,
                    cue_sign=0):

    trial_input, run_result = generate_test_trial(model_dir, hp, batch_size=batch_size,
                                                      rule_name=rule_name,
                                                      gaussian_center1=gaussian_center1,
                                                      gaussian_center2=gaussian_center2,
                                                      cue_sign=cue_sign)

    # print('select_cell',select_cell)
    if select_cell is None:
        firing_rate_cue = run_result.firing_rate_binder_pfc.detach().cpu().numpy()[start_proj:end_proj, :, :]
    else:
        firing_rate_cue = run_result.firing_rate_binder_pfc.detach().cpu().numpy()[start_proj:end_proj, :,select_cell]
    # firing_rate_list = list(firing_rate_cue[:, i, :] for i in range(0, batch_size))
    # concate_fr = np.concatenate(firing_rate_list, axis=0)
    # # print('concate_fr',concate_fr.shape)

    return firing_rate_cue








def generate_for_pca_parietal(model_dir,hp,batch_size=0,start_proj=0,end_proj=0,
                    rule_name='retro',
                    select_cell = None,
                    gaussian_center1=0,
                    gaussian_center2=0,
                    cue_sign=0):

    trial_input, run_result = generate_test_trial(model_dir, hp, batch_size=batch_size,
                                                      rule_name=rule_name,
                                                      gaussian_center1=gaussian_center1,
                                                      gaussian_center2=gaussian_center2,
                                                      cue_sign=cue_sign)
    if select_cell is None:
        firing_rate_cue = run_result.firing_rate_binder_parietal.detach().cpu().numpy()[start_proj:end_proj, :, :]
    else:
        firing_rate_cue = run_result.firing_rate_binder_parietal.detach().cpu().numpy()[start_proj:end_proj, :, select_cell]
    firing_rate_list = list(firing_rate_cue[:, i, :] for i in range(0, batch_size))
    concate_fr = np.concatenate(firing_rate_list, axis=0)
    # print('concate_fr',concate_fr.shape,concate_fr)

    return concate_fr



def get_allangle_activity_mean_space(model_dir,hp,batch_size=1,
                                rule_name='both',
                                cue_sign=None):


    if cue_sign is None:
        cue_sign = hp['rng'].choice([1,-1], (batch_size,))
    else:
        cue_sign = hp['rng'].choice([cue_sign], (batch_size,))


    # print('cue_sign',cue_sign)
    # sys.exit(0)




    pref = np.arange(0, 2 * np.pi, 2 * np.pi / hp['n_angle'])  # preferences
    fr_pfc_angles = []
    fr_parietal_angles = []
    fr_all_angles = []


    for angle in np.array(pref):

        gaussian_center1 = angle
        stim_dist = np.random.choice([1 * np.pi / 2, -1 * np.pi / 2])
        gaussian_center2 = (gaussian_center1 + stim_dist) % (2 * np.pi)#(gaussian_center1 + 1 * np.pi)% (2 * np.pi)

        runnerObj = run.Runner(rule_name=rule_name, hp=hp, model_dir=model_dir, is_cuda=False, noise_on=True, mode='test')
        trial_input, run_result = runnerObj.run(batch_size=batch_size,
                                                gaussian_center1=gaussian_center1,
                                                gaussian_center2=gaussian_center2,
                                                cue_sign=cue_sign)
        #### average value over batch_sizes for hidden state
        fr_pfc = run_result.firing_rate_binder_pfc.detach().cpu().numpy()
        fr_pfc_list = list([fr_pfc[:, i, :] for i in range(batch_size)])
        fr_pfc_mean = np.mean(np.array(fr_pfc_list), axis=0)
        fr_pfc_angles.append(fr_pfc_mean)

        fr_parietal = run_result.firing_rate_binder_parietal.detach().cpu().numpy()
        fr_parietal_list = list([fr_parietal[:, i, :] for i in range(batch_size)])
        fr_parietal_mean = np.mean(np.array(fr_parietal_list), axis=0)
        fr_parietal_angles.append(fr_parietal_mean)

        fr_all = run_result.firing_rate_binder.detach().cpu().numpy()
        fr_all_list = list([fr_all[:, i, :] for i in range(batch_size)])
        fr_all_mean = np.mean(np.array(fr_all_list), axis=0)
        fr_all_angles.append(fr_all_mean)


    fr_pfc_angles_array = np.array(fr_pfc_angles)
    fr_pfc_angles_mean=np.mean(np.array(fr_pfc_angles_array), axis=0)

    fr_parietal_angles_array = np.array(fr_parietal_angles)
    fr_parietal_angles_mean = np.mean(np.array(fr_parietal_angles_array), axis=0)


    fr_all_angles_array = np.array(fr_all_angles)
    fr_all_angles_mean = np.mean(np.array(fr_all_angles_array), axis=0)

    return fr_pfc_angles_mean,fr_parietal_angles_mean,fr_all_angles_mean




def find_encode_neuron_pfc(figure_path,model_dir, hp):
    batch_size = 512
    epoch = get_epoch(model_dir=model_dir, rule_name='retro', hp=hp)
    stim2_on = epoch['stim2_on'][0];
    stim2_off = epoch['stim2_off'][0]
    stim1_on = epoch['stim1_on'][0];
    stim1_off = epoch['stim1_off'][0];
    response_on = epoch['response_on'][0]
    response_off = epoch['response_off'][0]
    # print('epoch_0', epoch_0)

    cue_on = stim2_on
    cue_off = stim2_off

    exc = int(1*hp['n_rnn'])
    start = epoch['stim2_on'][0]
    end = epoch['stim2_off'][0]

    # start = epoch['stim2_on'][0] + 10
    # end = epoch['stim2_off'][0] + 20
    #
    firing_rate_up_ret,_,_ = get_allangle_activity_mean_space(model_dir, hp, batch_size=batch_size, rule_name='retro', cue_sign=1)
    firing_rate_down_ret,_,_ = get_allangle_activity_mean_space(model_dir, hp, batch_size=batch_size, rule_name='retro', cue_sign=-1)

    threshold = 0.02 * np.max(np.mean(firing_rate_up_ret[start:end, :], axis=0))
    print('threshold:',threshold)

    mean_up = np.where(np.mean(firing_rate_up_ret[start:end, :exc], axis=0) > threshold )[0]
    mean_down = np.where(np.mean(firing_rate_down_ret[start:end, :exc], axis=0) > threshold )[0]

    print('mean_up', mean_up.shape)
    print('mean_down', mean_down.shape)
    num = np.min([mean_up.shape[0],mean_down.shape[0]])



    return mean_up[:num],mean_down[:num]




def fit_data2(points):
    """
    Fit a plane to 3D points using least squares.
    Input:
        points: shape (N, 3), rows are (x, y, z)
    Returns:
        grid_x, grid_y, grid_z: plane surface (not used here)
        a, b, c: normal vector (a, b, c) of the plane ax + by + cz = d
    """
    from sklearn.linear_model import LinearRegression

    X = points[:, :2]  # x, y
    Z = points[:, 2]   # z

    reg = LinearRegression().fit(X, Z)
    a, b = reg.coef_
    # normal vector of plane: [-a, -b, 1] if z = ax + by + c
    normal = np.array([-a, -b, 1])
    return None, None, None, normal[0], normal[1], normal[2]


import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_planes_at_time(figure_path,points_up, points_down, normal_up, normal_down, time_idx, angle):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    sns.color_palette("Set2")
    _alpha_list = sns.color_palette("Set2")
    # Plot the 3D points
    print('points_up',points_up.shape)
    for i in range(points_up.shape[0]):
        ax.scatter(points_up[i, 0], points_up[i, 1], points_up[i, 2], color=_alpha_list[i], label='UP', s=60, marker='o')
        ax.scatter(points_down[i, 0], points_down[i, 1], points_down[i, 2], color=_alpha_list[i], label='DOWN', s=60, marker='^')

    # Create meshgrid for plane
    def plot_plane(points, normal, color):
        point = np.mean(points, axis=0)
        d = -point.dot(normal)
        xx, yy = np.meshgrid(
            np.linspace(point[0]-1, point[0]+1, 10),
            np.linspace(point[1]-1, point[1]+1, 10)
        )
        zz = (-normal[0] * xx - normal[1] * yy - d) / normal[2]
        ax.plot_surface(xx, yy, zz, alpha=0.3, color='grey', rstride=1, cstride=1, edgecolor='none',linewidth=0)



    # Plot the two planes

    plot_plane(points_up, normal_up, color='blue')
    plot_plane(points_down, normal_down, color='red')

    # Style
    fs = 10
    ax.set_xlabel('PC3', fontsize=fs + 5, labelpad=-5)
    ax.set_ylabel('PC2', fontsize=fs + 5, labelpad=-5)
    ax.set_zlabel('PC1', fontsize=fs + 5, labelpad=-5)
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    frame1.axes.zaxis.set_ticklabels([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(b=True)
    ax.xaxis._axinfo['grid']['color'] = (0.8, 0.8, 0.8, 0.5)  # light grey (R, G, B, alpha)
    ax.yaxis._axinfo['grid']['color'] = (0.8, 0.8, 0.8, 0.5)
    ax.zaxis._axinfo['grid']['color'] = (0.8, 0.8, 0.8, 0.5)

    ax.set_title('Time='+ str(time_idx)+'; angle='+str(angle))
    plt.savefig(figure_path + str(time_idx) + '_'  + '.pdf')
    plt.show()

def compute_plane_angle_each_timepoint(figure_path,concate_fr_a_list, concate_fr_b_list):
    """
    Compute the angle between planes (UP vs DOWN) at each timepoint.

    Parameters:
        concate_fr_a_list: list of 3 arrays (one per angle), each of shape (T, B, N)
        concate_fr_b_list: same as above, for cue_sign = -1

    Returns:
        time_angles: list of plane angles (in degrees) at each timepoint
    """
    T, B, N = concate_fr_a_list[0].shape
    num_angles = len(concate_fr_a_list)

    time_angles = []
    cosines=[]

    for t in range(T):
        # (1) collect PCA input from both conditions at time t
        all_data_t = []

        for a in concate_fr_a_list:
            all_data_t.append(a[t])  # shape (B, N)
        for b in concate_fr_b_list:
            all_data_t.append(b[t])

        all_data_t = np.concatenate(all_data_t, axis=0)  # shape (6*B, N)
        pca = PCA(n_components=3)
        pca.fit(all_data_t)

        # (2) get 3D means for each angle group in UP
        points_up = []
        for a in concate_fr_a_list:
            proj = pca.transform(a[t])  # (B, 3)
            points_up.append(np.mean(proj, axis=0))  # (3,)
        points_up = np.stack(points_up, axis=0)  # (3, 3)

        # (3) get 3D means for each angle group in DOWN
        points_down = []
        for b in concate_fr_b_list:
            proj = pca.transform(b[t])  # (B, 3)
            points_down.append(np.mean(proj, axis=0))
        points_down = np.stack(points_down, axis=0)  # (3, 3)

        # (4) fit planes and extract normal vectors
        _, _, _, a0, b0, c0 = fit_data2(points_up)
        _, _, _, a1, b1, c1 = fit_data2(points_down)

        n0 = np.array([a0, b0, c0])
        n1 = np.array([a1, b1, c1])

        cosine = np.dot(n0, n1) / (np.linalg.norm(n0) * np.linalg.norm(n1))
        cosine = np.clip(cosine, -1.0, 1.0)  # stability
        cosines.append(np.abs(cosine))

        # (5) compute angle
        dot = np.dot(n0, n1)
        norm0 = np.linalg.norm(n0)
        norm1 = np.linalg.norm(n1)
        angle_rad = np.arccos(np.clip(dot / (norm0 * norm1), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        if angle_deg > 90:
            angle_deg = 180 - angle_deg

        time_angles.append(angle_deg)
        print('angle_deg:',angle_deg)


        ############## plot 3D

        plot_planes_at_time(figure_path,points_up, points_down, n0, n1, time_idx=t, angle = angle_deg)



    return np.array(time_angles),cosines




def PCA_plot_3D_selected_angle_pfc_example1(fig_path, data_path,model_name, model_dir, idx, hp, rule_name,
                                start_proj=0,
                                end_proj=0):
    suffix = tools.splid_model_name(model_name, start_str='pc')
    file_name = suffix + '_' + str(idx) + '_seed' + str(hp['seed'])

    figure_path = os.path.join(fig_path, 'PCA_plot_3D_selected_angle_pfc_ins_paper' + str(hp['in_strength']) + '/')
    tools.mkdir_p(figure_path)
    data_path_0 = os.path.join(data_path, 'PCA_plot_3D_selected_angle_pfc' + '/')
    tools.mkdir_p(data_path_0)



    # encode_up_cell, encode_down_cell = find_encode_neuron_pfc(fig_path, model_dir, hp)
    # np.save(data_path_0 + file_name + '_encode_up_cell.npy', encode_up_cell)
    # np.save(data_path_0 + file_name + '_encode_down_cell.npy', encode_down_cell)

    encode_up_cell   = np.load(data_path_0 + file_name + '_encode_up_cell.npy')
    encode_down_cell = np.load(data_path_0 + file_name + '_encode_down_cell.npy')


    hp['rule_name'] = rule_name
    batch_size = 512

    prefs = np.arange(0, 2 * np.pi, 2 * np.pi / hp['n_angle'])  # preferences

    select_point = [0,3,6]
    pref_random = [prefs[select_point[0]], prefs[select_point[1]], prefs[select_point[2]]]
    pref = list(pref_random)
    # print('pref', pref)

    concate_fr_a_list = []
    concate_fr_b_list = []

    num_angle =len(pref)

    stim_dist = 1 * np.pi / 2#np.random.choice([1 * np.pi / 2, -1 * np.pi / 2])
    for i in range(num_angle):

        concate_fr_a0 = generate_for_pca_pfc(model_dir, hp, batch_size=batch_size, start_proj=start_proj, end_proj=end_proj,
                                         select_cell=encode_up_cell,
                                         gaussian_center1=pref[i],
                                         gaussian_center2=(pref[i] + stim_dist) % (2 * np.pi),
                                         cue_sign=1)


        concate_fr_b0 = generate_for_pca_pfc(model_dir, hp, batch_size=batch_size, start_proj=start_proj, end_proj=end_proj,
                                         select_cell=encode_down_cell,
                                         gaussian_center1=pref[i],
                                         gaussian_center2=(pref[i] + stim_dist) % (2 * np.pi),
                                         cue_sign=-1)
        print('concate_fr_a0',concate_fr_a0.shape)

        concate_fr_a_list.append(concate_fr_a0)#concate_fr_a0 = (3,10)
        concate_fr_b_list.append(concate_fr_b0)



    angles,cosines = compute_plane_angle_each_timepoint(figure_path,concate_fr_a_list, concate_fr_b_list)
    np.save(figure_path+'cosines.npy',cosines)

    cosines = np.load(figure_path+'cosines.npy')[:-1]
    print('cosines=',cosines)

    ################# fitting

    ################# fitting

    x = np.arange(len(cosines))  # Timepoints

    # --- Define the logistic function ---
    def logistic(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    # --- Initial parameter guess: (L=max, k=slope, x0=midpoint) ---
    p0 = [1.2, 1.0, np.median(x)]

    # Set bounds: L must be >1, k in a reasonable range, x0 within domain
    bounds = ([1.05, 0.01, min(x)], [2.0, 10.0, max(x)])

    # Fit with bounds
    popt, _ = curve_fit(logistic, x, cosines, p0=p0, bounds=bounds)
    L_fit, k_fit, x0_fit = popt

    # Generate fitted curve
    x_fit = np.linspace(min(x), max(x), 200)
    y_fit = logistic(x_fit, *popt)




    # # Logistic function definition
    # # Your data (replace with your actual data)
    # x = np.arange(20)
    # y = np.array([0.32, 0.14, 0.18, 0.15, 0.17, 0.21, 0.28, 0.31, 0.45, 0.80,
    #               0.99, 1.00, 1.00, 1.00, 0.90, 0.92, 0.95, 0.97, 0.98, 1.00])  # example
    #
    # # Define logistic function
    # def logistic(x, L, k, x0):
    #     return L / (1 + np.exp(-k * (x - x0)))
    #
    # # Initial guess: L=max(y), x0=center, k=slope
    # p0 = [1.0, 1.0, np.median(x)]
    # # Curve fitting
    # popt, _ = curve_fit(logistic, x, y, p0=p0)
    # # Generate fit curve
    # x_fit = np.linspace(min(x), max(x), 200)
    # y_fit = logistic(x_fit, *popt)



    fig = plt.figure(figsize=(2.0, 3))
    ax = fig.add_axes([0.25, 0.2, 0.7, 0.6])
    plt.title(file_name,fontsize=5)

    #axs.axvline(10, color='darkgrey', linestyle='--', label='stim_on')
    ax.plot(cosines, 'o', color='blue',markerfacecolor='white',markeredgewidth=1,markersize=3.5)
    plt.plot(x_fit, y_fit, 'b-',linewidth=1, label='Logistic fit')

    #plt.plot(cosines[:-1],'o',markersize=4)
    plt.xlabel("Timepoint")
    plt.ylabel("cosines of angle between plane")
    #plt.title("cosines value between upper and lower",fontsize=10)
    ax.axvspan(10, 10+2,color='darkgrey')
    ax.spines[['right', 'top']].set_visible(False)
    plt.ylim([0,1.03])
    plt.yticks([0,0.2,0.4,0.6,0.8,1.0])
    plt.savefig(figure_path + 'cosine_' + file_name+'_paper.pdf')

    plt.show()

