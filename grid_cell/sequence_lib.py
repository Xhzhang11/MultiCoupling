from sklearn.decomposition import PCA

import sys, os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib import cm
import matplotlib as mpl
import scipy
from scipy import io,interpolate
from scipy.interpolate import make_interp_spline
import cv2
from scipy import signal
import math
import seaborn as sns
import torch
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"../..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"../")))


from place_cells import PlaceCells



#
# #load parames
# hp = get_default_hp(random_seed=10000)
# arg = Key_to_value(hp)
#
# hp['Np'] = 1024
# hp['rng'] = np.random.RandomState(1)
# hp['run_ID'] = generate_run_ID(arg)
# hp['sequence_length'] = hp['seq_length_analysis']
#
#
#
# # img
# place_cells = PlaceCells(hp)
# trajectory_generator = TrajectoryGenerator(hp=hp, place_cells=place_cells)
#



def Evaluate_performance(hp,trajectory_generator,place_cells,model,fig_path,speed_scale):
    inputs, init_actv, pos, pc_outputs = trajectory_generator.get_batch_for_test(speed_scale)

    _,pred_activity = model(inputs, init_actv)

    pred_pos = place_cells.get_nearest_cell_pos(pred_activity)
    err = np.sqrt(((pos - pred_pos) ** 2).sum(-1)).mean() * 100

    print('error', err)

    us = place_cells.c_recep_field

    ss = 30  # 20
    # batch_size=2000,sl=30:7,8,9, 10,13, 21', 33,34, 35',36',37', 41,42,43,
    fig1 = plt.figure(figsize=(5, 5))
    ax = fig1.add_subplot(111)
    for j in range(0, 40):
        plt.plot(pos[:, j, 0], pos[:, j, 1], label='True position', c='red', linewidth=2.5)
        plt.plot(pred_pos[:, j, 0], pred_pos[:, j, 1], '.-', c='blue', label=str(j), linewidth=1, alpha=0.7,zorder=3)
        plt.plot(us[:, 0], us[:, 1], 'o', markersize=3, c='lightgrey', alpha=1)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)

    plt.xlim([-hp['box_width'] / 2, hp['box_width'] / 2])
    plt.ylim([-hp['box_height'] / 2, hp['box_height'] / 2])
    plt.title(str(hp['model_idx']))

    plt.savefig(fig_path + 'Evaluate_performance' + str(hp['model_idx']) +'_ls'+str(hp['sequence_length'])+ '.png')

    plt.show()



def ratemap_one_units(hp,trajectory_generator,model,unit_idx,speed_scale=0):
    #figure_path = os.path.join(fig_path, 'ratemap_one_units/'+str(hp['seq_length_analysis']))
    figure_path = os.path.join(fig_path, 'fig1/')
    mkdir_p(figure_path)

    res = 20#
    n_avg = 1#这个是可视化轨迹的关键

    activations, rate_map_lores, _, _ = compute_ratemaps(model, trajectory_generator, hp,
                                                         speed_scale=speed_scale,
                                                         res=res,
                                                         Ng=hp['Ng'],
                                                         n_avg=n_avg
                                                         )

    #model_9:452
    #17,505,435,372,134,136,306,311,312,365,381,403,414,473,492,497,508
    for i in unit_idx:#range(512):#E = 9,17,211,217;I=492,486
        # if i>=3:
        #     sys.exit(0)
        print('i',i)
        #print('activate',activations.shape,activations)

        im = activations[i,:,:]

        print('im',im.shape)
        im = (im - np.min(im)) / (np.max(im) - np.min(im))# normalization
        cmap = plt.cm.get_cmap('jet')#

        np.seterr(invalid='ignore')  # ignore divide by zero err
        im = cv2.GaussianBlur(im, (3,3), sigmaX=1, sigmaY=0)
        im = cmap(im)
        # im = np.uint8(im * 255)
        im = im[:,:,0]


        ############## plot hidden
        fig = plt.figure(figsize=(1, 1))
        #im = (im - np.min(im))/(np.max(im)-np.min(im))
        #plt.matshow(im,vmin=0.1, vmax=0.9)
        plt.matshow(im,vmin=0.1, vmax=0.9)
        plt.title('unit_'+str(i)+';'+'model_'+str(hp['seq_length_model'])+';'+
                  'speed_'+str(speed_scale)+';'+'sl_'+str(hp['seq_length_analysis'])+';'+
                  'visual_'+str(hp['vis_input']))
        plt.axis('off')
        plt.savefig(figure_path+'/'+'unit_'+str(i)+'.png')
        plt.show()

        #autocorrelation
        """
        fig_correlation = plt.figure(figsize=(1, 1))
        in1 = im
        in2 = im
        im = signal.correlate(in1, in2, mode='same', method='direct')
        im = (im - np.min(im))/(np.max(im)-np.min(im))
        print('**im',im.shape,np.min(im),np.max(im))
        plt.matshow(im,vmin=0.0, vmax=0.5)
        plt.title('unit_'+str(i)+';'+'model_'+str(hp['seq_length_model'])+';'+
                  'speed_'+str(speed_scale)+';'+'sl_'+str(hp['seq_length_analysis'])+';'+
                  'visual_'+str(hp['vis_input']))
        plt.axis('off')
        plt.savefig(figure_path+'/correlation_unit_'+str(i)+'.png')
        plt.show()
        """

        ###

        #================================= calculate the grid score ===============================


def Calculate_grid_score(hp,trajectory_generator,model,fig_path):
    figure_path = os.path.join(fig_path, 'Calculate_grid_score/'+'score_60')
    mkdir_p(figure_path)
    res = 50
    n_avg = 1

    low_res = 20
    starts = [0.2] * 10
    ends = np.linspace(0.4, 0.8, num=10)#np.linspace(0.4, 1.0, num=10)
    box_width=hp['box_width']
    box_height=hp['box_height']
    coord_range=((-box_width/2, box_width/2), (-box_height/2, box_height/2))
    masks_parameters = zip(starts, ends.tolist())
    scorer = GridScorer(low_res, coord_range, masks_parameters)


    # Compute a set of low-res maps to use for evalutaing grid score
    ######################################################################################################

    _, rate_map_lores, _, _ = visualize.compute_ratemaps(model, trajectory_generator, hp,
                                                         res=low_res,
                                                         Ng=hp['Ng'],
                                                         n_avg=n_avg)

    score_60, score_90, sac, max_60_ind = zip(*[scorer.get_scores(rm.reshape(low_res, low_res)) for rm in tqdm(rate_map_lores)])
    score_type = score_60
    io.savemat(figure_path+'/'+"score_sl_"+str(hp['seq_length_analysis'])+".mat",{'score':score_type})
    ######################################################################################################

    load_data = io.loadmat(figure_path+'/'+"score_sl_"+str(hp['seq_length_analysis'])+".mat")
    score_type = load_data['score'][0,:]

    idxs = np.flip(np.argsort(score_type))
    score_type = score_type[idxs]
    print('score_type',np.round(score_type,3))

    # #Plot high grid scores
    # k = -1
    # for idx in range(512):
    #     k += 1
    #     im = activations[idx,:,:]
    #     image = visualize.rgb(im, smooth=True)#(50,50,4)
    #     ############## plot hidden
    #     fig = plt.figure(figsize=(1, 1))
    #     plt.matshow(image[:,:,0])
    #     plt.title('unit_'+str(idx)+'; score:' +str(np.round(score_type[k],2)))
    #     plt.axis('off')
    #     plt.savefig(figure_path+'/'+str(k)+'.png')
    #     plt.show()





def Plot_ratemap(hp,trajectory_generator,model,scale,fig_path):

    figure_path = os.path.join(fig_path, hp['get_grid'])
    mkdir_p(figure_path)
    res = 50
    n_avg = 2
    activations, _, _, _ = visualize.compute_ratemaps(model, trajectory_generator, hp,
                                                      res=res,
                                                      Ng=hp['Ng'],
                                                      n_avg=n_avg)
    # print(get_grid,'activations',activations[0,:10,:10])

    ######Compute a set of low-res maps to use for evalutaing grid score
    #### each unit###########
    for idx in range(2):
        fig1 = plt.figure(figsize=(5, 5))
        ax = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
        plt.title(hp['get_grid'] + '_' + str(idx)+'; '+str(hp['degree'])+'_'+str(scale), fontsize=15)

        # print(get_grid, 'activations', activations[idx])
        im = activations[idx, :, :]
        im = (im - np.min(im)) / (np.max(im) - np.min(im))  # normalization
        im = cv2.GaussianBlur(im, (3, 3), sigmaX=1, sigmaY=0)
        ax.imshow(im, interpolation='none', cmap='jet');
        ax.axis('off')

        plt.savefig(figure_path+'/'+str(idx)+hp['get_grid']+'_'+str(hp['degree'])+'_'+str(scale)+hp['env']+'.pdf')
        plt.show()


def plot_trajectory(hp,pos,pred_pos):
    fig1 = plt.figure(figsize=(5, 5))
    ax1 = fig1.add_subplot(111)

    colors = sns.color_palette()  # sns.color_palette("Set2")
    i=0
    for k1 in range(30,40):

        plt.plot(pos[:, k1, 0], pos[:, k1, 1], c='black', linewidth=1)

        # ax1.scatter(pos[0,k1,0], pos[0,k1,1],   marker='o', color='red')
        # ax1.scatter(pos[-1,k1,0], pos[-1,k1,1],   marker='*', color='red')

        plt.plot(pred_pos[:, k1, 0], pred_pos[:, k1, 1], '.-', c=colors[i], label=str(k1), linewidth=1)
        # ax1.scatter(pred_pos[0,k1,0], pred_pos[0,k1,1],   marker='o', color='b')
        # ax1.scatter(pred_pos[-2,k1,0], pred_pos[-2,k1,1],   marker='*', color='b')
        plt.xlim([-hp['box_width'] / 2, hp['box_width'] / 2])
        plt.ylim([-hp['box_height'] / 2, hp['box_height'] / 2])
        plt.title(hp['get_grid']+'; length='+str(hp['sequence_length']))

        plt.legend()
        for axis in ['top', 'bottom', 'left', 'right']:
            ax1.spines[axis].set_linewidth(2)
        # plt.savefig(figure_path+'/'+str(k1)+'.png')
        i += 1

    plt.show()



def get_information_speedscale(hp,trajectory_generator,model,get_grid,speed_scale):
    place_cells = PlaceCells(hp)
    inputs_0, init_actv_0, pos_0, _ = trajectory_generator.get_batch_for_test(speed_scale=speed_scale,
                                                                              batch_size=hp['batch_size_test'])

    if hp['get_grid'] == 'all':
        _, pred_activity_binder_0 = model(inputs_0, init_actv_0)
    else:
        pred_activity_binder_0, _ = model(inputs_0, init_actv_0)

    pred_pos_0 = place_cells.get_nearest_cell_pos(pred_activity_binder_0)

    pred_activity_binder_0 = pred_activity_binder_0.detach().cpu().numpy()
    mean_fr_0 = np.mean(pred_activity_binder_0, axis=1)

    fig1 = plt.figure(figsize=(5, 5))
    ax1 = fig1.add_subplot(111)

    colors = sns.color_palette()  # sns.color_palette("Set2")
    i = 0
    for k1 in range(0, 40):

        plt.plot(pos_0[:, k1, 0], pos_0[:, k1, 1], c='black', linewidth=1)

        # ax1.scatter(pos[0,k1,0], pos[0,k1,1],   marker='o', color='red')
        # ax1.scatter(pos[-1,k1,0], pos[-1,k1,1],   marker='*', color='red')

        plt.plot(pred_pos_0[:, k1, 0], pred_pos_0[:, k1, 1], '.-', label=str(k1), linewidth=1)
        # ax1.scatter(pred_pos[0,k1,0], pred_pos[0,k1,1],   marker='o', color='b')
        # ax1.scatter(pred_pos[-2,k1,0], pred_pos[-2,k1,1],   marker='*', color='b')
        plt.xlim([-hp['box_width'] / 2, hp['box_width'] / 2])
        plt.ylim([-hp['box_height'] / 2, hp['box_height'] / 2])
        plt.title(hp['get_grid'] + '; length=' + str(hp['sequence_length'])+'; speed=' + str(speed_scale))

        plt.legend()
        for axis in ['top', 'bottom', 'left', 'right']:
            ax1.spines[axis].set_linewidth(2)
        # plt.savefig(figure_path+'/'+str(k1)+'.png')
        i += 1

    plt.show()

    return pos_0, pred_pos_0, mean_fr_0


def plot_neuron_activity(hp,trajectory_generator,model,fig_path):
    figure_path = os.path.join(fig_path, 'neuron_activity/'+str(hp['seq_length_analysis']))
    mkdir_p(figure_path)

    get_grid = hp['get_grid']


    ########################################### speed =0 ######################################################
    pos_0, pred_pos_0, mean_fr_0 = get_information_speedscale(hp, trajectory_generator, model, get_grid=get_grid,
                                                              speed_scale=0)
    plot_trajectory(hp, pos_0, pred_pos_0)

    ########################################### speed =1 ######################################################

    pos_1, pred_pos_1, mean_fr_1 = get_information_speedscale(hp, trajectory_generator, model, get_grid=get_grid,
                                                              speed_scale=1)
    plot_trajectory(hp, pos_1, pred_pos_1)




    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    for i in range(1000):
        plt.plot(mean_fr_0[:, i])

    plt.title(hp['get_grid'])
    plt.ylabel('Firing rate (a.u.)')
    plt.xlabel('Time (ms)')
    plt.tick_params(axis='both')
    # plt.ylim([-0.05,0.12])
    # plt.savefig(figure_path+'/'+ str(unit_idx)+'.png')
    plt.show()



    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    for i in range(1000):
        plt.plot(mean_fr_1[:, i])

    plt.title(hp['get_grid'])

    plt.ylabel('Firing rate (a.u.)')
    plt.xlabel('Time (ms)')
    plt.tick_params(axis='both')
    # plt.ylim([-0.05,0.12])
    # plt.savefig(figure_path+'/'+ str(unit_idx)+'.png')
    plt.show()

def plot_neuron_activity_combine(hp,trajectory_generator,model,fig_path):
    figure_path = os.path.join(fig_path, 'neuron_activity/'+str(hp['seq_length_analysis']))
    mkdir_p(figure_path)

    get_grid = hp['get_grid']



    ########################################### speed =0 ######################################################
    # hp['sequence_length'] = 10
    pos_0, pred_pos_0, mean_fr_0 = get_information_speedscale(hp, trajectory_generator, model, get_grid=get_grid,
                                                              speed_scale=0)

    ########################################### speed =1 ######################################################
    # hp['sequence_length'] = 10
    pos_1, pred_pos_1, mean_fr_1 = get_information_speedscale(hp, trajectory_generator, model, get_grid=get_grid,
                                                              speed_scale=1)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4),layout='constrained', sharex=True, sharey=True)
    fig.suptitle(hp['get_grid']+'; test_length='+str(hp['sequence_length']))
    for i in range(1000):
        axs[0].plot(mean_fr_0[:, i])
        axs[1].plot(mean_fr_1[:, i])

    axs[0].set_title('speed=0')
    axs[1].set_title('speed=1')

    plt.ylabel('Firing rate (a.u.)')
    plt.xlabel('Time (ms)')
    plt.tick_params(axis='both')
    # plt.ylim([-0.05,0.12])
    # plt.savefig(figure_path+'/'+ str(unit_idx)+'.png')
    plt.show()

def plot_neuron_activity_single(hp,trajectory_generator,model,fig_path):
    figure_path = os.path.join(fig_path, 'neuron_activity/'+str(hp['seq_length_analysis']))
    mkdir_p(figure_path)

    get_grid = hp['get_grid']



    ########################################### speed =0 ######################################################

    pos_0, pred_pos_0, mean_fr_0 = get_information_speedscale(hp, trajectory_generator, model, get_grid=get_grid,
                                                              speed_scale=1)
    plot_trajectory(hp, pos_0, pred_pos_0)






    # fig, axs = plt.subplots(1, 2, figsize=(10, 4),layout='constrained', sharex=True, sharey=True)
    # fig.suptitle(hp['get_grid']+'; test_length='+str(hp['sequence_length']))
    # for i in range(1000):
    #     axs[0].plot(mean_fr_0[:, i])
    # axs[0].set_title('speed=0')
    # plt.ylabel('Firing rate (a.u.)')
    # plt.xlabel('Time (ms)')
    # plt.tick_params(axis='both')
    # # plt.ylim([-0.05,0.12])
    # # plt.savefig(figure_path+'/'+ str(unit_idx)+'.png')
    # plt.show()

def plot_sorting(hp,trajectory_generator,model,fig_path,speed_scale):
    figure_path = os.path.join(fig_path, 'neuron_activity/'+str(hp['seq_length_analysis']))
    mkdir_p(figure_path)

    get_grid = hp['get_grid']



    ########################################### speed =0 ######################################################
    speed_scale=speed_scale
    pos_0, pred_pos_0, mean_fr_0 = get_information_speedscale(hp, trajectory_generator, model, get_grid=get_grid,
                                                              speed_scale=speed_scale)





    # plot_trajectory(hp, pos_0, pred_pos_0)

    data_0 = mean_fr_0
    func_activity_threshold_0=0.0001

    print('mean_fr_0',mean_fr_0.shape)
    max_firing_rate_vis = np.max(data_0, axis=0)
    print('max_firing_rate_vis',max_firing_rate_vis.shape,max_firing_rate_vis)
    pick_idx_vis = np.argwhere(max_firing_rate_vis > func_activity_threshold_0).squeeze()

    data_0 = data_0[:, pick_idx_vis]
    peak_time_vis = np.argmax(data_0, axis=0)

    peak_order_vis = np.argsort(peak_time_vis, axis=0)
    data_0 = data_0[:, peak_order_vis]

    for i in range(0, data_0.shape[1]):
        peak = np.max(data_0[:, i])
        if peak < 0:
            data_0[:, i] = 0
        else:
            data_0[:, i] = data_0[:, i] / peak
            # print("np.max(data[:, i])",np.max(data[:, i]))
    # '''
    # Prepare grid for plotting

    data_0 = data_0[1:,:]
    time_step = 1
    X_0, Y_0 = np.mgrid[0:data_0.shape[0] * time_step:time_step, 0:data_0.shape[1]]

    fig = plt.figure(figsize=(3.2, 2.5))
    ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])

    fs = 15

    # Make the plot
    # cmap = plt.get_cmap('viridis')#viridis_r
    plt.pcolormesh(X_0, Y_0, data_0)
    m = cm.ScalarMappable(cmap=mpl.rcParams["image.cmap"])  # cmap=mpl.rcParams["image.cmap"]
    m.set_array([0, 1])
    m.set_clim(vmin=0, vmax=1)
    cbar = fig.colorbar(m, ax=ax, aspect=15)
    cbar.set_ticks([0, 1])
    cbar.ax.tick_params(labelsize=fs)

    file_name=get_grid+'; sl='+str(hp['sequence_length'])+';speed='+str(speed_scale)
    plt.title(file_name,fontsize=8)

    plt.savefig(fig_path +  file_name+'_sorting.pdf')
    plt.show()






    # fig, axs = plt.subplots(1, 2, figsize=(10, 4),layout='constrained', sharex=True, sharey=True)
    # fig.suptitle(hp['get_grid']+'; test_length='+str(hp['sequence_length']))
    # for i in range(1000):
    #     axs[0].plot(mean_fr_0[:, i])
    # axs[0].set_title('speed=0')
    # plt.ylabel('Firing rate (a.u.)')
    # plt.xlabel('Time (ms)')
    # plt.tick_params(axis='both')
    # # plt.ylim([-0.05,0.12])
    # # plt.savefig(figure_path+'/'+ str(unit_idx)+'.png')
    # plt.show()
def plot_sorting_find_cell(hp,trajectory_generator,model,fig_path,speed_scale):
    figure_path = os.path.join(fig_path, 'plot_sorting_find_cell/ls_'+str(hp['sequence_length']))
    mkdir_p(figure_path)

    get_grid = hp['get_grid']

    ########################################### speed =0 ######################################################
    speed_scale = speed_scale
    place_cells = PlaceCells(hp)
    inputs_0, init_actv_0, pos_0, _ = trajectory_generator.get_batch_for_test(speed_scale=speed_scale,
                                                                              batch_size=hp['batch_size_test'])

    if hp['get_grid'] == 'all':
        _, pred_activity_binder_0 = model(inputs_0, init_actv_0)
    else:
        pred_activity_binder_0, _ = model(inputs_0, init_actv_0)

    pred_pos_0 = place_cells.get_nearest_cell_pos(pred_activity_binder_0)

    pred_activity_binder_0 = pred_activity_binder_0.detach().cpu().numpy()
    mean_fr_0 = np.mean(pred_activity_binder_0, axis=1)

    data_0 = mean_fr_0
    func_activity_threshold_0 = 0.0001

    print('mean_fr_0', mean_fr_0.shape)
    max_firing_rate_vis = np.max(data_0, axis=0)
    print('max_firing_rate_vis', max_firing_rate_vis.shape, max_firing_rate_vis)
    pick_idx_vis = np.argwhere(max_firing_rate_vis > func_activity_threshold_0).squeeze()

    data_0 = data_0[:, pick_idx_vis]
    peak_time_vis = np.argmax(data_0, axis=0)

    peak_order_vis = np.argsort(peak_time_vis, axis=0)
    selected_cell = peak_order_vis[0:1500]
    data_0 = data_0[:, selected_cell]

    for i in range(0, data_0.shape[1]):
        peak = np.max(data_0[:, i])
        data_0[:, i] = data_0[:, i] / peak


    # Prepare grid for plotting
    data_0 = data_0[1:, :]
    time_step = 1
    X_0, Y_0 = np.mgrid[0:data_0.shape[0] * time_step:time_step, 0:data_0.shape[1]]

    fig = plt.figure(figsize=(3.2, 2.5))
    ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])

    fs = 15

    # Make the plot
    # cmap = plt.get_cmap('viridis')#viridis_r
    plt.pcolormesh(X_0, Y_0, data_0)
    m = cm.ScalarMappable(cmap=mpl.rcParams["image.cmap"])  # cmap=mpl.rcParams["image.cmap"]
    m.set_array([0, 1])
    m.set_clim(vmin=0, vmax=1)
    cbar = fig.colorbar(m, ax=ax, aspect=15)
    cbar.set_ticks([0, 1])
    cbar.ax.tick_params(labelsize=fs)

    file_name = get_grid + ';sl=' + str(hp['sequence_length']) + ';speed=' + str(speed_scale)
    plt.title(file_name, fontsize=8)
    plt.savefig(fig_path + file_name + '_sorting.pdf')
    plt.show()



    ################### place cell ###################
    ################### place cell ###################
    ################### place cell ###################
    colors = sns.dark_palette("#69d", reverse=True, as_cmap=True)#sns.color_palette("magma", as_cmap=True)#sns.cubehelix_palette(start=2, rot=-0.5, dark=0, light=1, reverse=True, as_cmap=True)

    num_points = 10
    from matplotlib.collections import LineCollection

    recep_field = place_cells.c_recep_field

    for group in range(512):
        # Select local segment of points
        local_cell = selected_cell[group:group + num_points]
        local_recep_field = recep_field[local_cell]

        # Create segments for gradient line
        gradient_line = local_recep_field.reshape(-1, 1, 2)
        segments = np.concatenate([gradient_line[:-1], gradient_line[1:]], axis=1)
        gradient = np.linspace(0, 1, num_points - 1)

        # Create LineCollection
        lc = LineCollection(segments, cmap='Blues', linewidth=1)
        lc.set_array(gradient)

        # Plot
        fig, ax = plt.subplots(figsize=(5, 5))
        # Add the gradient line
        ax.add_collection(lc)
        # Scatter individual points
        for j, cell in enumerate(local_cell):
            ax.plot(recep_field[cell, 0], recep_field[cell, 1],
                    'o', c='red', markersize=8, markeredgecolor='grey', label=str(j))


        ax.set_title(f'Group {group}')
        plt.xlim([-hp['box_width'] / 2, hp['box_width'] / 2])
        plt.ylim([-hp['box_height'] / 2, hp['box_height'] / 2])
        # plt.legend()





        # i = 0
        # for k1 in range(0, 40):
        #     plt.plot(pos_0[:, k1, 0], pos_0[:, k1, 1], c='black', linewidth=1)
        #     #plt.plot(pred_pos_0[:, k1, 0], pred_pos_0[:, k1, 1], '.-', label=str(k1),markersize=1, linewidth=0.5)
        #
        #     plt.title(hp['get_grid'] + '; length=' + str(hp['sequence_length']) + '; speed=' + str(speed_scale))
        #
        #     #plt.legend()
        #     for axis in ['top', 'bottom', 'left', 'right']:
        #         ax.spines[axis].set_linewidth(2)
        #
        #     i += 1

        plt.savefig(figure_path+'/'+str(speed_scale)+'_'+str(group)+'.png')
        # plt.show()


def plot_sorting_find_cell_select(hp,trajectory_generator,model,fig_path,speed_scale):
    figure_path = os.path.join(fig_path, 'plot_sorting_find_cell/ls_'+str(hp['sequence_length']))
    mkdir_p(figure_path)

    get_grid = hp['get_grid']

    ########################################### speed =0 ######################################################
    speed_scale = speed_scale
    place_cells = PlaceCells(hp)
    inputs_0, init_actv_0, pos_0, _ = trajectory_generator.get_batch_for_test(speed_scale=speed_scale,
                                                                              batch_size=hp['batch_size_test'])

    if hp['get_grid'] == 'all':
        _, pred_activity_binder_0 = model(inputs_0, init_actv_0)
    else:
        pred_activity_binder_0, _ = model(inputs_0, init_actv_0)

    pred_pos_0 = place_cells.get_nearest_cell_pos(pred_activity_binder_0)

    pred_activity_binder_0 = pred_activity_binder_0.detach().cpu().numpy()

    print(' pred_activity_binder_0', pred_activity_binder_0.shape)
    mean_fr_0 = np.mean(pred_activity_binder_0[:,:,:], axis=1)

    data_0 = mean_fr_0
    func_activity_threshold_0 = 0.0002

    print('mean_fr_0', mean_fr_0.shape)
    max_firing_rate_vis = np.max(data_0, axis=0)
    print('max_firing_rate_vis', max_firing_rate_vis.shape, max_firing_rate_vis)
    pick_idx_vis = np.argwhere(max_firing_rate_vis > func_activity_threshold_0).squeeze()

    data_0 = data_0[:, pick_idx_vis]
    peak_time_vis = np.argmax(data_0, axis=0)

    peak_order_vis = np.argsort(peak_time_vis, axis=0)
    selected_cell = peak_order_vis[:]
    data_0 = data_0[:, selected_cell]

    for i in range(0, data_0.shape[1]):
        peak = np.max(data_0[:, i])
        data_0[:, i] = data_0[:, i] / peak


    # Prepare grid for plotting
    data_0 = data_0[1:, :]
    time_step = 1
    X_0, Y_0 = np.mgrid[0:data_0.shape[0] * time_step:time_step, 0:data_0.shape[1]]



    fig = plt.figure(figsize=(3.2, 2.5))
    ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
    # Make the plot
    # cmap = plt.get_cmap('viridis')#viridis_r
    plt.pcolormesh(X_0, Y_0, data_0)
    m = cm.ScalarMappable(cmap=mpl.rcParams["image.cmap"])  # cmap=mpl.rcParams["image.cmap"]
    m.set_array([0, 1])
    m.set_clim(vmin=0, vmax=1)
    cbar = fig.colorbar(m, ax=ax, aspect=15)
    cbar.set_ticks([0, 1])
    cbar.ax.tick_params(labelsize=15)
    file_name = get_grid + ';sl=' + str(hp['sequence_length']) + ';speed=' + str(speed_scale)
    plt.title(file_name, fontsize=8)
    # plt.yticks([0,20,40,60,80,100,120,140,160,180])
    # plt.yticks([0, 10,20, 30,40, 50,60, 70,80, 90,100,])
    plt.savefig(fig_path + file_name + '_sorting.pdf')
    plt.show()

    print('==========')


    sys.exit(0)



    ################### place cell ###################
    ################### place cell ###################
    ################### place cell ###################
    colors = sns.color_palette()  # sns.color_palette("Set2")

    num_points = 10
    from matplotlib.collections import LineCollection

    recep_field = place_cells.c_recep_field



    # Select local segment of points
    # idxs = [70,100,150,198,202]
    idxs = [20,55,56,57,58,]#range(10)

    local_cell = selected_cell[idxs]
    local_recep_field = recep_field[local_cell]

    # Create segments for gradient line
    gradient_line = local_recep_field.reshape(-1, 1, 2)
    segments = np.concatenate([gradient_line[:-1], gradient_line[1:]], axis=1)
    gradient = np.linspace(0, 1, num_points - 1)

    # Create LineCollection
    lc = LineCollection(segments, cmap='Blues', linewidth=1)
    lc.set_array(gradient)

    # Plot
    fig, ax = plt.subplots(figsize=(5, 5))
    # Add the gradient line
    ax.add_collection(lc)
    # Scatter individual points
    for j, cell in enumerate(local_cell):
        ax.plot(recep_field[cell, 0], recep_field[cell, 1],
                'o', c=colors[j], markersize=8, markeredgecolor='grey', label=str(j),zorder=3)

    us = place_cells.c_recep_field
    plt.plot(us[:, 0], us[:, 1], 'o', markersize=3, c='lightgrey', alpha=1)

    plt.xlim([-hp['box_width'] / 2, hp['box_width'] / 2])
    plt.ylim([-hp['box_height'] / 2, hp['box_height'] / 2])
    plt.legend()





    # i = 0
    # for k1 in range(0, 40):
    #     plt.plot(pos_0[:, k1, 0], pos_0[:, k1, 1], c='black', linewidth=1)
    #     #plt.plot(pred_pos_0[:, k1, 0], pred_pos_0[:, k1, 1], '.-', label=str(k1),markersize=1, linewidth=0.5)
    #
    #     plt.title(hp['get_grid'] + '; length=' + str(hp['sequence_length']) + '; speed=' + str(speed_scale))
    #
    #     #plt.legend()
    #     for axis in ['top', 'bottom', 'left', 'right']:
    #         ax.spines[axis].set_linewidth(2)
    #
    #     i += 1

    plt.savefig(figure_path+'/'+str(speed_scale)+'_'+str(idxs)+'.png')
    plt.show()

def plot_activity_heatmap_for_each_trajectory_speed1(hp,trajectory_generator,model,fig_path,speed_scale):
    figure_path = os.path.join(fig_path, 'plot_activity_heatmap_for_each_trajectory_speed1/ls_'+str(hp['sequence_length'])
                               +'tau'+str(hp['tau'])+'speed_'+str(speed_scale))
    mkdir_p(figure_path)

    get_grid = hp['get_grid']

    ########################################### speed =0 ######################################################
    speed_scale = speed_scale
    place_cells = PlaceCells(hp)
    inputs_0, init_actv_0, pos_0, _ = trajectory_generator.get_batch_for_test(speed_scale=speed_scale,
                                                                              batch_size=hp['batch_size_test'])

    if hp['get_grid'] == 'all':
        _, pred_activity_binder_0 = model(inputs_0, init_actv_0)
    else:
        pred_activity_binder_0, _ = model(inputs_0, init_actv_0)

    pred_pos_0,traj_identity = place_cells.get_nearest_cell_pos_find_sequence(pred_activity_binder_0)


    # print('traj_identity',traj_identity.shape)
    #


    pred_activity_binder_0 = pred_activity_binder_0.detach().cpu().numpy()

    # print('pred_activity_binder_0',pred_activity_binder_0.shape)
    # mean_fr_0 = np.mean(pred_activity_binder_0, axis=1)




    #for j in np.array([4,69,96,122,148,205,293,303,356,367,411,412,464,471,496]):
    for j in range(5):

        print(j)


        traj_select = traj_identity[1:,j]

        # traj_select = torch.unique(traj_select)
        data_0 = pred_activity_binder_0[1:,j,:]

        # Prepare grid for plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5),gridspec_kw={'width_ratios': [1.2, 1]} )  # 1 row, 2 columns
        fig.suptitle(str(hp['sequence_length'])+'tau'+str(hp['tau'])+'_model'+str(hp['model_idx'])+'; traj='+str(j), fontsize=7)

        # ---------- Left plot: pcolormesh ----------
        mesh = ax1.pcolormesh(data_0.T, shading='auto', cmap='viridis')
        cbar = fig.colorbar(mesh, ax=ax1, aspect=15)  # 使用 mesh 对象来生成 colorbar
        cbar.ax.tick_params(labelsize=10)
        ax1.set_xlabel('trajectory')
        ax1.set_ylabel('plece cell')

        # print(data_0[:,350:600])

        max_firing_rate_vis = np.max(data_0, axis=0)

        # print('max_firing_rate_vis',max_firing_rate_vis.shape)
        func_activity_threshold_0=0.0045
        pick_idx_vis = np.argwhere(max_firing_rate_vis > func_activity_threshold_0).squeeze()
        # print('pick_idx_vis',pick_idx_vis)




        # ---------- Right plot: place cells and predicted trajectory ----------
        us = place_cells.c_recep_field
        ax2.plot(pred_pos_0[1:, j, 0], pred_pos_0[1:, j, 1], '.-', c='blue', label=f'Traj {j}', linewidth=1, alpha=0.9,zorder=3)
        ax2.plot(pred_pos_0[0, j, 0], pred_pos_0[0, j, 1], 'o', c='red', label=f'Traj {j}', markersize=1.5, alpha=0.9,zorder=4)
        ax2.plot(us[:, 0], us[:, 1], 'o', markersize=1, c='lightgrey', alpha=0.5, label='Place cells')

        for cell in np.array(pick_idx_vis):
            ax2.plot(us[cell, 0], us[cell, 1], 'x', c='grey',markersize=5, alpha=1, label=str(cell))

        # ax2.set_xlim([-hp['box_width'] / 2, hp['box_width'] / 2])
        # ax2.set_ylim([-hp['box_height'] / 2, hp['box_height'] / 2])
        ax2.set_title(f"Model {hp['model_idx']}", fontsize=10)
        #ax2.legend(fontsize=3)

        plt.tight_layout()
        plt.savefig(figure_path + '/' + str(speed_scale) + '_' + str(j) + '.png')
        #plt.savefig(figure_path + '/' + str(speed_scale) + '_' + str(j) + '.pdf')

        # plt.show()

def plot_heatmap_for_each_trajectory(hp,trajectory_generator,model,fig_path,speed_scale):
    figure_path = os.path.join(fig_path, 'plot_activity_heatmap_for_each_trajectory/ls_'+str(hp['sequence_length'])
                               +'tau'+str(hp['tau'])+'speed_'+str(speed_scale))
    mkdir_p(figure_path)

    get_grid = hp['get_grid']

    ########################################### speed =0 ######################################################
    speed_scale = speed_scale
    place_cells = PlaceCells(hp)
    inputs_0, init_actv_0, pos_0, _ = trajectory_generator.get_batch_for_test(speed_scale=speed_scale,
                                                                              batch_size=hp['batch_size_test'])

    if hp['get_grid'] == 'all':
        _, pred_activity_binder_0 = model(inputs_0, init_actv_0)
    else:
        pred_activity_binder_0, _ = model(inputs_0, init_actv_0)

    pred_pos_0,traj_identity = place_cells.get_nearest_cell_pos_find_sequence(pred_activity_binder_0)


    # print('traj_identity',traj_identity.shape)
    #


    pred_activity_binder_0 = pred_activity_binder_0.detach().cpu().numpy()

    # print('pred_activity_binder_0',pred_activity_binder_0.shape)
    # mean_fr_0 = np.mean(pred_activity_binder_0, axis=1)




    #for j in np.array([4,69,96,122,148,205,293,303,356,367,411,412,464,471,496]):
    for j in range(512):

        print(j)


        traj_select = traj_identity[1:,j]

        # traj_select = torch.unique(traj_select)
        data_0 = pred_activity_binder_0[0:,j,:]

        # Prepare grid for plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5),gridspec_kw={'width_ratios': [1.2, 1]} )  # 1 row, 2 columns
        fig.suptitle(str(hp['sequence_length'])+'_tau'+str(hp['tau'])
                     +'_model'+str(hp['model_idx'])+';    traj='+ str(j), fontsize=15)

        # ---------- Left plot: pcolormesh ----------
        mesh = ax1.pcolormesh(data_0.T, shading='auto', cmap='viridis')
        cbar = fig.colorbar(mesh, ax=ax1, aspect=15)  # 使用 mesh 对象来生成 colorbar
        cbar.ax.tick_params(labelsize=10)
        # ax1.set_xlabel('trajectory')
        # ax1.set_ylabel('plece cell')

        # print(data_0[:,350:600])

        max_firing_rate_vis = np.max(data_0, axis=0)

        # print('max_firing_rate_vis',max_firing_rate_vis.shape)
        func_activity_threshold_0=0.0001
        pick_idx_vis = np.argwhere(max_firing_rate_vis > func_activity_threshold_0).squeeze()
        # print('pick_idx_vis',pick_idx_vis)




        # ---------- Right plot: place cells and predicted trajectory ----------
        us = place_cells.c_recep_field
        ax2.plot(pred_pos_0[0:, j, 0], pred_pos_0[0:, j, 1], '.-', c='blue', label=f'Traj {j}', linewidth=1, alpha=0.9,zorder=3)
        ax2.plot(pred_pos_0[0, j, 0], pred_pos_0[0, j, 1], 'o', c='red', label=f'Traj {j}', markersize=5, alpha=0.9,zorder=4)
        ax2.plot(us[:, 0], us[:, 1], 'o', markersize=1, c='lightgrey', alpha=0.5, label='Place cells')

        for cell in np.array(pick_idx_vis):
            ax2.plot(us[cell, 0], us[cell, 1], 'x', c='grey',markersize=5, alpha=1, label=str(cell))

        # ax2.set_xlim([-hp['box_width'] / 2, hp['box_width'] / 2])
        # ax2.set_ylim([-hp['box_height'] / 2, hp['box_height'] / 2])
        ax2.set_title('position='+str(pred_pos_0[0, j, :]), fontsize=10)
        #ax2.legend(fontsize=3)
        # ax2.axis('off')

        plt.tight_layout()
        plt.savefig(figure_path + '/' + str(speed_scale) + '_' + str(j) + '.png')
        #plt.savefig(figure_path + '/' + str(speed_scale) + '_' + str(j) + '.pdf')

        # plt.show()

def plot_heatmap_for_each_trajectory_diff_seed(hp,trajectory_generator,model,fig_path,speed_scale):
    figure_path = os.path.join(fig_path, 'plot_heatmap_for_each_trajectory_diff_seed')
    mkdir_p(figure_path)

    get_grid = hp['get_grid']

    ########################################### speed =0 ######################################################
    speed_scale = speed_scale
    place_cells = PlaceCells(hp)
    inputs_0, init_actv_0, pos_0, _ = trajectory_generator.get_batch_for_test(speed_scale=speed_scale,
                                                                              batch_size=hp['batch_size_test'])

    if hp['get_grid'] == 'all':
        _, pred_activity_binder_0 = model(inputs_0, init_actv_0)
    else:
        pred_activity_binder_0, _ = model(inputs_0, init_actv_0)

    pred_pos_0,traj_identity = place_cells.get_nearest_cell_pos_find_sequence(pred_activity_binder_0)


    pred_activity_binder_0 = pred_activity_binder_0.detach().cpu().numpy()

    # print('pred_activity_binder_0',pred_activity_binder_0.shape)
    # mean_fr_0 = np.mean(pred_activity_binder_0, axis=1)




    #for j in np.array([4,69,96,122,148,205,293,303,356,367,411,412,464,471,496]):
    for j in np.array([43,24,202,279,276,357,490]):

        print(j)
        file_name = 'ls' + str(hp['sequence_length']) + '_traj' + str(j) + '_seed' + str(hp['seed'])

        traj_select = traj_identity[1:,j]

        # traj_select = torch.unique(traj_select)
        data_0 = pred_activity_binder_0[2:,j,:]

        # Prepare grid for plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5),gridspec_kw={'width_ratios': [1.2, 1]} )  # 1 row, 2 columns
        fig.suptitle(str(hp['sequence_length'])+'_tau'+str(hp['tau'])
                     +'_model'+str(hp['model_idx'])+'; traj='+ str(j)+'; seed='+ str(hp['seed']), fontsize=15)

        # ---------- Left plot: pcolormesh ----------
        mesh = ax1.pcolormesh(data_0.T, shading='auto', cmap='viridis')
        cbar = fig.colorbar(mesh, ax=ax1, aspect=15)  # 使用 mesh 对象来生成 colorbar
        cbar.ax.tick_params(labelsize=10)
        # ax1.set_xlabel('trajectory')
        # ax1.set_ylabel('plece cell')

        # print(data_0[:,350:600])

        max_firing_rate_vis = np.max(data_0, axis=0)

        # print('max_firing_rate_vis',max_firing_rate_vis.shape)
        func_activity_threshold_0=0.0045
        pick_idx_vis = np.argwhere(max_firing_rate_vis > func_activity_threshold_0).squeeze()
        # print('pick_idx_vis',pick_idx_vis)




        # ---------- Right plot: place cells and predicted trajectory ----------
        us = place_cells.c_recep_field
        ax2.plot(pred_pos_0[0:, j, 0], pred_pos_0[0:, j, 1], '.-', c='blue', label=f'Traj {j}', linewidth=1, alpha=0.9,zorder=3)
        ax2.plot(pred_pos_0[0, j, 0], pred_pos_0[0, j, 1], 'o', c='red', label=f'Traj {j}', markersize=5, alpha=0.9,zorder=4)
        ax2.plot(us[:, 0], us[:, 1], 'o', markersize=1, c='lightgrey', alpha=0.5, label='Place cells')

        for cell in np.array(pick_idx_vis):
            ax2.plot(us[cell, 0], us[cell, 1], 'x', c='grey',markersize=5, alpha=1, label=str(cell))

        # ax2.set_xlim([-hp['box_width'] / 2, hp['box_width'] / 2])
        # ax2.set_ylim([-hp['box_height'] / 2, hp['box_height'] / 2])
        ax2.set_title('position='+str(pred_pos_0[0, j, :]), fontsize=10)
        #ax2.legend(fontsize=3)
        # ax2.axis('off')

        plt.tight_layout()
        plt.savefig(figure_path + '/' + file_name + '.png')
        #plt.savefig(figure_path + '/' + str(speed_scale) + '_' + str(j) + '.pdf')

        # plt.show()

def plot_activity_heatmap_for_each_trajectory_onlyheatmap(hp,trajectory_generator,model,fig_path,speed_scale):
    figure_path = os.path.join(fig_path, 'plot_activity_heatmap_for_each_trajectory_onlyheatmap/ls_'+str(hp['sequence_length'])+'tau'+str(hp['tau']))
    mkdir_p(figure_path)

    get_grid = hp['get_grid']

    ########################################### speed =0 ######################################################
    speed_scale = speed_scale
    place_cells = PlaceCells(hp)
    inputs_0, init_actv_0, pos_0, _ = trajectory_generator.get_batch_for_test(speed_scale=speed_scale,
                                                                              batch_size=hp['batch_size_test'])

    if hp['get_grid'] == 'all':
        _, pred_activity_binder_0 = model(inputs_0, init_actv_0)
    else:
        pred_activity_binder_0, _ = model(inputs_0, init_actv_0)

    pred_pos_0,traj_identity = place_cells.get_nearest_cell_pos_find_sequence(pred_activity_binder_0)


    # print('traj_identity',traj_identity.shape)
    #


    pred_activity_binder_0 = pred_activity_binder_0.detach().cpu().numpy()

    # print('pred_activity_binder_0',pred_activity_binder_0.shape)
    # mean_fr_0 = np.mean(pred_activity_binder_0, axis=1)




    for j in range(4):
        print(j)


        traj_select = traj_identity[1:,j]

        # traj_select = torch.unique(traj_select)
        data_0 = pred_activity_binder_0[1:,j,:]


        fig = plt.figure(figsize=(3.2, 2.5))
        ax1 = fig.add_axes([0.15, 0.1, 0.7, 0.8])
        title_name = str(hp['sequence_length'])+'tau'+str(hp['tau'])+'_model'+str(hp['model_idx'])+'_traj='+str(j)
        plt.title(title_name, fontsize=7)

        # ---------- Left plot: pcolormesh ----------
        mesh = ax1.pcolormesh(data_0.T, shading='auto', cmap='viridis')
        cbar = fig.colorbar(mesh, ax=ax1, aspect=15)  # 使用 mesh 对象来生成 colorbar
        cbar.ax.tick_params(labelsize=10)
        ax1.set_xlabel('trajectory')
        ax1.set_ylabel('plece cell')

        # print(data_0[:,350:600])

        plt.tight_layout()
        plt.savefig(figure_path + '/' +str(speed_scale) + '_' + str(j) + '.png')
        #plt.savefig(figure_path + '/' + str(speed_scale) + '_' + str(j) + '.pdf')

        plt.show()


def plot_sorting_find_cell_select_method4(hp,trajectory_generator,model,fig_path,speed_scale):
    figure_path = os.path.join(fig_path, 'plot_sorting_find_cell_select_method3/ls_'+str(hp['sequence_length']))
    mkdir_p(figure_path)

    get_grid = hp['get_grid']

    ########################################### speed =0 ######################################################
    speed_scale = speed_scale
    place_cells = PlaceCells(hp)
    inputs_0, init_actv_0, pos_0, _ = trajectory_generator.get_batch_for_test(speed_scale=speed_scale,
                                                                              batch_size=hp['batch_size_test'])

    if hp['get_grid'] == 'all':
        _, pred_activity_binder_0 = model(inputs_0, init_actv_0)
    else:
        pred_activity_binder_0, _ = model(inputs_0, init_actv_0)

    pred_pos_0,traj_identity = place_cells.get_nearest_cell_pos_find_sequence(pred_activity_binder_0)


    print('traj_identity',traj_identity.shape)
    #


    pred_activity_binder_0 = pred_activity_binder_0.detach().cpu().numpy()

    print('pred_activity_binder_0',pred_activity_binder_0.shape)
    # mean_fr_0 = np.mean(pred_activity_binder_0, axis=1)




    for j in np.array([412,]):#4,303,367,411,412,496
        print(j,'===============================')

        traj_select = traj_identity[2:,j]
        data_0 = pred_activity_binder_0[2:,j,40:400]
        threshold=0.005


        all_activate_cell_list = []

        for i in range(12):
            get_activate_cell =  np.argwhere(data_0[i,:] > threshold).squeeze()
            all_activate_cell_list.append(get_activate_cell)


        print('all_activate_cell_list',all_activate_cell_list)

        # Ensure all elements are at least 1D arrays
        flat_list = np.concatenate([np.atleast_1d(x) for x in all_activate_cell_list if x.size > 0])
        unique_cells = np.unique(flat_list)
        # Extract columns (cells) that are active at any time point
        select_data = data_0[:, unique_cells]



        print('unique_cells',unique_cells)




        # Prepare grid for plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.5),gridspec_kw={'width_ratios': [1.4, 1]} )  # 1 row, 2 columns
        title_name ='_model'+str(hp['model_idx'])+'; tau'+str(hp['tau'])+'; ls='+str(hp['sequence_length'])+'; traj='+str(j)+ '; threshold_'+str(threshold)
        fig.suptitle(title_name, fontsize=12)

        # ---------- Left plot: pcolormesh ----------
        mesh = ax1.pcolormesh(select_data.T, shading='auto', cmap='viridis')
        cbar = fig.colorbar(mesh, ax=ax1, aspect=15)  # 使用 mesh 对象来生成 colorbar
        cbar.ax.tick_params(labelsize=10)

        ax1.set_xlabel('trajectory')
        ax1.set_ylabel('plece cell')
        # ax1.set_ylim([0,350])

        # print(data_0[:,350:600])



        data_1 = pred_activity_binder_0[2:, j, :]
        max_firing_rate_vis = np.max(data_1, axis=0)
        #print('max_firing_rate_vis',max_firing_rate_vis.shape)
        func_activity_threshold_0=threshold
        pick_idx_vis = np.argwhere(max_firing_rate_vis > func_activity_threshold_0).squeeze()
        #print('pick_idx_vis',pick_idx_vis)

        # ---------- Right plot: place cells and predicted trajectory ----------
        us = place_cells.c_recep_field
        ax2.plot(pred_pos_0[0:, j, 0], pred_pos_0[0:, j, 1], '.-', c='blue', label=f'Traj {j}', linewidth=1, alpha=0.9,
                 zorder=3)

        ax2.plot(pred_pos_0[0, j, 0], pred_pos_0[0, j, 1], '*', c='green', label=f'Traj {j}', linewidth=1, alpha=0.9,
                 zorder=3)

        ax2.plot(us[:, 0], us[:, 1], 'o', markersize=1, c='lightgrey', alpha=0.5, label='Place cells')

        for cell in np.array(pick_idx_vis):
            ax2.plot(us[cell, 0], us[cell, 1], 'x', markersize=5, alpha=1, label=str(cell))

        # ax2.set_xlim([-hp['box_width'] / 2, hp['box_width'] / 2])
        # ax2.set_ylim([-hp['box_height'] / 2, hp['box_height'] / 2])

        # ax2.set_xlim([-0.4, 0.2])
        # ax2.set_ylim([-0.4, 0.3])
        ax2.set_title(f"Model {hp['model_idx']}", fontsize=10)
        # ax2.legend(fontsize=3)

        plt.tight_layout()
        plt.savefig(figure_path + '/' + str(speed_scale) + '_' + str(j) + '.png')
        plt.show()




        ######################## fig2 #########################################
        ######################## fig2 #########################################
        ######################## fig2 #########################################

        # colors = sns.color_palette()  # sns.color_palette("Set2")
        #
        # fig1 = plt.figure(figsize=(5, 5))
        # ax = fig1.add_subplot(111)
        # ss=8
        #
        #
        #
        # cell_group1 = [396,397,398,399]
        # cell_group2 = [428,429,430,431]
        # cell_group3 = [460, 461, 462, 463,464]
        # cell_group4 = [492, 493, 494, 495, 496]
        # cell_group5 = [525,526,527,528]
        # cell_group6 = [557, 558,559]
        # cell_group7 = [589,590]
        #
        # pick_idx_vis=cell_group1+cell_group2+cell_group3+cell_group4+cell_group5+cell_group6+cell_group7
        #
        #
        # for cell in np.array(cell_group1):
        #     ax.plot(us[cell, 0], us[cell, 1], 'x', c = colors[0],markersize=ss, alpha=1, label=str(cell))
        #
        # for cell in np.array(cell_group2):
        #     ax.plot(us[cell, 0], us[cell, 1], 'x', c=colors[1], markersize=ss, alpha=1, label=str(cell))
        #
        # for cell in np.array(cell_group3):
        #     ax.plot(us[cell, 0], us[cell, 1], 'x', c = colors[2],markersize=ss, alpha=1, label=str(cell))
        #
        # for cell in np.array(cell_group4):
        #     ax.plot(us[cell, 0], us[cell, 1], 'x', c=colors[3], markersize=ss, alpha=1, label=str(cell))
        # for cell in np.array(cell_group5):
        #     ax.plot(us[cell, 0], us[cell, 1], 'x', c=colors[4], markersize=ss, alpha=1, label=str(cell))
        # for cell in np.array(cell_group6):
        #     ax.plot(us[cell, 0], us[cell, 1], 'x', c=colors[5], markersize=ss, alpha=1, label=str(cell))
        #
        # for cell in np.array(cell_group7):
        #     ax.plot(us[cell, 0], us[cell, 1], 'x', c=colors[6], markersize=ss, alpha=1, label=str(cell))
        #
        # ax.plot(us[:, 0], us[:, 1], 'o', markersize=1, c='lightgrey', alpha=0.5, label='Place cells')
        # ax.plot(pred_pos_0[0:, j, 0], pred_pos_0[0:, j, 1], '.-', c='black', label=f'Traj {j}', linewidth=1, alpha=0.3,
        #         zorder=3)
        #
        # ax.set_xlim([-0.4, 0.2])
        # ax.set_ylim([-0.4, 0.3])
        # ax.legend(fontsize=3)
        # plt.show()
        #
        #
        #
        #
        #
        # #################
        #
        # cell_choose = [430,429,462,494,526,558]
        # fig1 = plt.figure(figsize=(5, 5))
        # ax = fig1.add_subplot(111)
        #
        # tmp=0
        #
        # for i in np.array(cell_choose):
        #     ax.plot(data_0[:, i],c=colors[tmp],label = str(i))
        #     tmp += 1
        # ax.set_title('speed='+str(speed_scale))
        # plt.ylabel('Firing rate (a.u.)')
        # plt.xlabel('Time (ms)')
        # plt.tick_params(axis='both')
        # ax.legend(fontsize=7)
        #
        # # plt.ylim([-0.05,0.12])
        # # plt.savefig(figure_path+'/'+ str(unit_idx)+'.png')
        # plt.show()



def plot_sorting_find_cell_select_method5(hp,trajectory_generator,model,fig_path,speed_scale):
    figure_path = os.path.join(fig_path, 'plot_sorting_find_cell_select_method3/ls_'+str(hp['sequence_length']))
    mkdir_p(figure_path)

    get_grid = hp['get_grid']

    ########################################### speed =0 ######################################################
    speed_scale = speed_scale
    place_cells = PlaceCells(hp)
    inputs_0, init_actv_0, pos_0, _ = trajectory_generator.get_batch_for_test(speed_scale=speed_scale,
                                                                              batch_size=hp['batch_size_test'])

    if hp['get_grid'] == 'all':
        _, pred_activity_binder_0 = model(inputs_0, init_actv_0)
    else:
        pred_activity_binder_0, _ = model(inputs_0, init_actv_0)

    pred_pos_0,traj_identity = place_cells.get_nearest_cell_pos_find_sequence(pred_activity_binder_0)


    print('traj_identity',traj_identity.shape)
    #


    pred_activity_binder_0 = pred_activity_binder_0.detach().cpu().numpy()

    print('pred_activity_binder_0',pred_activity_binder_0.shape)
    # mean_fr_0 = np.mean(pred_activity_binder_0, axis=1)




    for j in np.array([4,303,367,411,412,496]):#4,303,367,411,412,496
        print(j,'===============================')

        data_0 = pred_activity_binder_0[2:,j,:]
        threshold=0.0049


        all_activate_cell_list = []

        for i in range(12):
            print('==== i',i)
            get_activate_cell =  np.argwhere(data_0[i,:] > threshold).flatten()
            print('get_activate_cell ',get_activate_cell)
            # if get_activate_cell.size!=0:
            all_activate_cell_list.append(get_activate_cell)


        print('all_activate_cell_list:',all_activate_cell_list)


        #print('all_activate_cell_list',all_activate_cell_list)

        # Ensure all elements are at least 1D arrays
        flat_list = np.concatenate([np.atleast_1d(x) for x in all_activate_cell_list if x.size > 0])
        unique_cells = np.unique(flat_list)

        #### fliter the cell
        select_cells = [i for i in unique_cells if 40<i<400]
        # Extract columns (cells) that are active at any time point
        select_data = data_0[:, select_cells]



        #print('unique_cells',unique_cells)
        #print('select_cells', select_cells)



        # Prepare grid for plotting
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.5),gridspec_kw={'width_ratios': [1.3,1.0, 1]} )  # 1 row, 2 columns
        title_name ='_model'+str(hp['model_idx'])+'; tau'+str(hp['tau'])+'; ls='+str(hp['sequence_length'])+'; traj='+str(j)+ '; threshold_'+str(threshold)
        fig.suptitle(title_name, fontsize=12)

        # ---------- Left plot: pcolormesh ----------
        mesh = ax1.pcolormesh(select_data.T, shading='auto', cmap='viridis')
        cbar = fig.colorbar(mesh, ax=ax1, aspect=15)  # 使用 mesh 对象来生成 colorbar
        cbar.ax.tick_params(labelsize=10)

        ax1.set_xlabel('trajectory')
        ax1.set_ylabel('plece cell')
        # ax1.set_ylim([0,350])





        cell_activity = [23, 57, 97, 120, 162, 214, 240, 309, 265, 361]
        mesh = ax2.pcolormesh(data_0[:, cell_activity].T, shading='auto', cmap='viridis')
        cbar = fig.colorbar(mesh, ax=ax2, aspect=15)  # 使用 mesh 对象来生成 colorbar
        cbar.ax.tick_params(labelsize=10)

        ax2.set_xlabel('trajectory')
        ax2.set_ylabel('plece cell')

        # print(data_0[:,350:600])



        # ---------- Right plot: place cells and predicted trajectory ----------
        # ---------- Right plot: place cells and predicted trajectory ----------
        # ---------- Right plot: place cells and predicted trajectory ----------
        us = place_cells.c_recep_field
        ax3.plot(pred_pos_0[0:, j, 0], pred_pos_0[0:, j, 1], '.-', c='blue', label=f'Traj {j}', linewidth=1, alpha=0.9,)
        ax3.plot(pred_pos_0[0, j, 0], pred_pos_0[0, j, 1], '*', c='green', label=f'Traj {j}', linewidth=1, alpha=0.9,)
        ax3.plot(us[:, 0], us[:, 1], 'o', markersize=1, c='lightgrey', alpha=0.5, label='Place cells')


        for cell in np.array(cell_activity):
            ax3.plot(us[cell, 0], us[cell, 1], 'x', markersize=8, alpha=1, label=str(cell),zorder=3)

        # ax2.set_xlim([-hp['box_width'] / 2, hp['box_width'] / 2])
        # ax2.set_ylim([-hp['box_height'] / 2, hp['box_height'] / 2])

        # ax2.set_xlim([-0.4, 0.2])
        # ax2.set_ylim([-0.4, 0.3])
        ax3.set_title(f"Model {hp['model_idx']}", fontsize=10)
        # ax2.legend(fontsize=3)

        plt.tight_layout()
        plt.savefig(figure_path + '/' + str(speed_scale) + '_' + str(j) + '.png')
        plt.show()




        ######################## fig2 #########################################
        ######################## fig2 #########################################
        ######################## fig2 #########################################

        colors = sns.color_palette()  # sns.color_palette("Set2")
        #
        # fig1 = plt.figure(figsize=(5, 5))
        # ax = fig1.add_subplot(111)
        # ss=8
        #
        #
        #
        # cell_group1 = [396,397,398,399]
        # cell_group2 = [428,429,430,431]
        # cell_group3 = [460, 461, 462, 463,464]
        # cell_group4 = [492, 493, 494, 495, 496]
        # cell_group5 = [525,526,527,528]
        # cell_group6 = [557, 558,559]
        # cell_group7 = [589,590]
        #
        # pick_idx_vis=cell_group1+cell_group2+cell_group3+cell_group4+cell_group5+cell_group6+cell_group7
        #
        #
        # for cell in np.array(cell_group1):
        #     ax.plot(us[cell, 0], us[cell, 1], 'x', c = colors[0],markersize=ss, alpha=1, label=str(cell))
        #
        # for cell in np.array(cell_group2):
        #     ax.plot(us[cell, 0], us[cell, 1], 'x', c=colors[1], markersize=ss, alpha=1, label=str(cell))
        #
        # for cell in np.array(cell_group3):
        #     ax.plot(us[cell, 0], us[cell, 1], 'x', c = colors[2],markersize=ss, alpha=1, label=str(cell))
        #
        # for cell in np.array(cell_group4):
        #     ax.plot(us[cell, 0], us[cell, 1], 'x', c=colors[3], markersize=ss, alpha=1, label=str(cell))
        # for cell in np.array(cell_group5):
        #     ax.plot(us[cell, 0], us[cell, 1], 'x', c=colors[4], markersize=ss, alpha=1, label=str(cell))
        # for cell in np.array(cell_group6):
        #     ax.plot(us[cell, 0], us[cell, 1], 'x', c=colors[5], markersize=ss, alpha=1, label=str(cell))
        #
        # for cell in np.array(cell_group7):
        #     ax.plot(us[cell, 0], us[cell, 1], 'x', c=colors[6], markersize=ss, alpha=1, label=str(cell))
        #
        # ax.plot(us[:, 0], us[:, 1], 'o', markersize=1, c='lightgrey', alpha=0.5, label='Place cells')
        # ax.plot(pred_pos_0[0:, j, 0], pred_pos_0[0:, j, 1], '.-', c='black', label=f'Traj {j}', linewidth=1, alpha=0.3,
        #         zorder=3)
        #
        # ax.set_xlim([-0.4, 0.2])
        # ax.set_ylim([-0.4, 0.3])
        # ax.legend(fontsize=3)
        # plt.show()
        #
        #
        #
        #
        #
        # #################
        #
        cell_choose = cell_activity
        fig1 = plt.figure(figsize=(5, 3))
        ax = fig1.add_subplot(111)

        tmp=0

        data_1 = pred_activity_binder_0[:,j,:]

        for i in np.array(cell_choose):
            ax.plot(data_1[:, i],c=colors[tmp],label = str(i))
            tmp += 1
        ax.set_title('speed='+str(speed_scale))
        plt.ylabel('Firing rate (a.u.)')
        plt.xlabel('Time (ms)')
        plt.tick_params(axis='both')
        ax.legend(fontsize=7)

        # plt.ylim([-0.05,0.12])
        # plt.savefig(figure_path+'/'+ str(unit_idx)+'.png')
        # plt.show()





def plot_sequence_and_cell_activity_speed0_tau25(data_path,hp,fig_path,speed_scale):
    figure_path = os.path.join(fig_path, 'plot_sequence_and_cell_activity_speed0_tau25')


    data_path = os.path.join(data_path, 'plot_sequence_and_cell_activity_speed0_tau25')


    get_grid = hp['get_grid']

    ########################################### speed =0 ######################################################
    speed_scale = speed_scale

    # #
    # inputs_0, init_actv_0, pos_0, _ = trajectory_generator.get_batch_for_test(speed_scale=speed_scale,
    #                                                                           batch_size=hp['batch_size_test'])
    #
    # if hp['get_grid'] == 'all':
    #     _, pred_activity_binder_0 = model(inputs_0, init_actv_0)
    # else:
    #     pred_activity_binder_0, _ = model(inputs_0, init_actv_0)
    #
    # pred_activity_binder_0 = pred_activity_binder_0.detach().cpu().numpy()
    # np.save(data_path + 'pred_activity_binder_sl'+str(hp['sequence_length'])+'tau'+str(hp['tau'])+'seed'+str(hp['seed'])+'.npy', pred_activity_binder_0)
    # #



    pred_activity_binder_0 = torch.tensor(np.load(data_path + 'pred_activity_binder_sl'+str(hp['sequence_length'])+'tau'+str(hp['tau'])+'seed'+str(hp['seed'])+'.npy'))
    place_cells = PlaceCells(hp)
    pred_pos_0,traj_identity = place_cells.get_nearest_cell_pos_find_sequence(pred_activity_binder_0)

    pred_activity_binder_0 = pred_activity_binder_0.detach().cpu().numpy()






    trajs =[24,43,202,276,279,357,490]# [24,43,202,276,279,357,490]#[33,74]#[24,43,202,276,279,357,490]
    trajs_reverse = [12,33,34,46,70,213,223]
    trajs_init_faraway = [42,140,394,271]#394,271

    for j in np.array(trajs_init_faraway):#4,303,367,411,412,496
        print(j,'===============================')
        fig_path_0 = os.path.join(figure_path, 'traj_faraway_'+str(j))



        threshold = 0.0047
        file_name = 'ls' + str(hp['sequence_length']) + '_traj' + str(j) + '_seed' + str(hp['seed'])
        title_name = '_model' + str(hp['model_idx']) + '; tau' + str(hp['tau']) + '; ls=' + str(
            hp['sequence_length']) + '; traj=' + str(j) + '; threshold_' + str(threshold) + '; seed=' + str(hp['seed'])

        # Example: extract and plot from trial j

        data_0 = pred_activity_binder_0[2:, j, :]

        all_activate_cell_list = []

        for i in range(hp['sequence_length'] - 2):
            get_activate_cell = np.argwhere(data_0[i, :] > threshold).flatten()

            mean_all = np.mean(get_activate_cell)
            if mean_all<200:
                min_cell = 0
                max_cell = 200
                all_activate_cell_list.append(get_activate_cell[::-1])
                reverse=True


            else:
                min_cell = 180
                max_cell = 500
                reverse = False
                all_activate_cell_list.append(get_activate_cell)

            print('get_activate_cell', get_activate_cell)
            # get_activate_cell
            all_activate_cell_list.append(get_activate_cell)

        # Ensure all elements are at least 1D arrays
        flat_list = np.concatenate([np.atleast_1d(x) for x in all_activate_cell_list if x.size > 0])
        unique_flat_list = list(dict.fromkeys(flat_list))

        if reverse:

            select_cells = [i for i in unique_flat_list if min_cell < i <= max_cell][::-1]
        else:
            select_cells = [i for i in unique_flat_list if min_cell < i <= max_cell]

        select_data = data_0[:, select_cells]



        # Prepare grid for plotting
        ####################### heatmap fig1 #########################################
        ####################### heatmap fig1 #########################################

        fig1 = plt.figure(figsize=(4, 3))
        ax1 = fig1.add_axes([0.15, 0.15, 0.7, 0.7])
        fig1.suptitle(title_name, fontsize=7)
        # ---------- Left plot: pcolormesh ----------
        mesh = ax1.pcolormesh(select_data.T, shading='auto', cmap='viridis')
        cbar = fig1.colorbar(mesh, ax=ax1, aspect=15)  # 使用 mesh 对象来生成 colorbar
        cbar.ax.tick_params(labelsize=10)

        ax1.set_xlabel('trajectory')
        ax1.set_ylabel('plece cell')
        # ax1.set_ylim([0,350])
        plt.show()
        #

        #
        # ######################## trajectory fig3 #########################################
        # ######################## trajectory fig3 #########################################
        # ######################## trajectory fig3 #########################################
        #
        fig1 = plt.figure(figsize=(3, 3))
        ax4 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])


        us = place_cells.c_recep_field
        ax4.plot(pred_pos_0[2:, j, 0], pred_pos_0[2:, j, 1], '.-', c='tab:orange', linewidth=1.5,alpha=1, )
        ax4.plot(pred_pos_0[0, j, 0], pred_pos_0[0, j, 1], 'o', c='green', markersize=4, alpha=1)

        ax4.plot(pred_pos_0[2, j, 0], pred_pos_0[2, j, 1], '*', c='blue', markersize=6, alpha=0.9, )
        ax4.plot(pred_pos_0[-1, j, 0], pred_pos_0[-1, j, 1], '>', c='blue', markersize=5, alpha=0.9, )

        plt.scatter(us[:,0], us[:,1], c='lightgrey', s=5,alpha=1,linewidths=None,edgecolors=None)
        plt.scatter([0], [0], c='black', marker='+', s=50, alpha=0.9,linewidths=None,edgecolors=None)

        outer_circle = plt.Circle((0, 0), hp['box_width'] / 2, fill=False, linestyle='-', linewidth=1,color='gray')
        ax4.add_patch(outer_circle)
        # for cell in np.array(cell_activity):
        #     #ax4.plot(us[cell, 0], us[cell, 1], 'x', markersize=5, alpha=1, label=str(cell), zorder=3)
        #
        #     # Small circle with radius 0.1
        #     select_point_x = place_cells.c_recep_field[cell, 0]
        #     select_point_y = place_cells.c_recep_field[cell, 1]
        #     small_circle = plt.Circle((select_point_x, select_point_y), 0.13, fill=False, color='blue', linewidth=1)
        #     #plt.plot(select_point_x, select_point_y, 'o', c='tab:orange', markersize=3)
        #     #ax4.add_patch(small_circle)
        ax4.set_title(file_name, fontsize=10)
        #ax4.legend(fontsize=5)
        ax4.axis('off')
        #plt.tight_layout()
        plt.show()
















        ######################## single activity fig4 #########################################
        ######################## single activity fig4 #########################################
        ######################## single activity fig4 #########################################
        #
        # colors = sns.color_palette()  # sns.color_palette("Set2")
        # cell_choose = cell_activity
        # fig1 = plt.figure(figsize=(5, 3))
        # ax = fig1.add_subplot(111)
        # tmp=0
        # data_1 = pred_activity_binder_0[:,j,:]
        #
        # for i in np.array(cell_choose):
        #     ax.plot(data_1[:, i],c=colors[tmp],label = str(i))
        #     tmp += 1
        # ax.set_title(title_name)
        # plt.ylabel('Firing rate (a.u.)')
        # plt.xlabel('Time (ms)')
        # plt.tick_params(axis='both')
        # ax.legend(fontsize=7)
        # # plt.ylim([-0.05,0.12])
        # # plt.savefig(figure_path+'/'+ str(unit_idx)+'.png')
        # plt.show()




















