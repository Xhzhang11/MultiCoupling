from sklearn.decomposition import PCA

import sys, os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import scipy
from scipy import io,interpolate
from scipy.interpolate import make_interp_spline

from scipy import signal
import math


sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"../..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"../")))





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



def Evaluate_performance(hp,trajectory_generator,model):
    figure_path = os.path.join(fig_path, 'Evaluate_performance/'+str(hp['seq_length_analysis']))
    mkdir_p(figure_path)


    inputs, pos, pc_outputs = trajectory_generator.get_batch_for_test()
    pred_activity = model.forward_predict(inputs)
    pred_pos = place_cells.get_nearest_cell_pos(pred_activity)
    err = np.sqrt(((pos - pred_pos)**2).sum(-1)).mean()*100

    us = place_cells.c_recep_field


    ss=30#20
    #batch_size=2000,sl=30:7,8,9, 10,13, 21', 33,34, 35',36',37', 41,42,43,

    for j in np.array([10]):
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        #plt.scatter(us[:,0], us[:,1], s=20, alpha=0.5, c='lightgrey')
        i=j#+5*11
        plt.plot(pos[:,i,0], pos[:,i,1], label='True position', c='black',linewidth=2.5)
        ax.scatter(pos[0,i,0], pos[0,i,1],   s=ss,marker='o', color='black')
        ax.scatter(pos[-1,i,0], pos[-1,i,1],  s=ss, marker='*', color='black')

        ax.plot(pred_pos[:-2,i,0], pred_pos[:-2,i,1], '-',c='red', label='Decoded position',linewidth=2,zorder=1)
        ax.plot(pred_pos[:-3,i,0], pred_pos[:-3,i,1], 'o',c='tab:blue', markersize=4, label='Decoded position')
        ax.scatter(pred_pos[0,i,0], pred_pos[0,i,1], s=ss,  marker='o', color='tab:blue')
        ax.scatter(pred_pos[-3,i,0], pred_pos[-3,i,1],  s=ss, marker='*', color='tab:blue',zorder=3)




    # for k1 in range(100):
    #     fig1 = plt.figure(figsize=(5,5))
    #     ax1 = fig1.add_subplot(111)
    #
    #     plt.plot(pos[:,k1,0], pos[:,k1,1], c='black', label='True position', linewidth=1)
    #     ax1.scatter(pos[0,k1,0], pos[0,k1,1],   marker='o', color='red')
    #     ax1.scatter(pos[-1,k1,0], pos[-1,k1,1],   marker='*', color='red')
    #
    #     plt.plot(pred_pos[:-1,k1,0], pred_pos[:-1,k1,1], '.-', c='tab:orange', label='Decoded position',linewidth=1)
    #     ax1.scatter(pred_pos[0,k1,0], pred_pos[0,k1,1],   marker='o', color='b')
    #     ax1.scatter(pred_pos[-2,k1,0], pred_pos[-2,k1,1],   marker='*', color='b')
    #     plt.xlim([-hp['box_width']/2,hp['box_width']/2])
    #     plt.ylim([-hp['box_height']/2,hp['box_height']/2])
    #     plt.title(k1)
    #     for axis in ['top','bottom','left','right']:
    #         ax1.spines[axis].set_linewidth(2)
    #     plt.savefig(figure_path+'/'+str(k1)+'.png')



    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
    plt.xticks([])
    plt.yticks([])
    plt.xlim([-0.9,0.1])
    plt.ylim([-0.2,0.7])
    plt.title('batch_size_test='+ str(hp['batch_size_test'])+' '+
              'seq_length_analysis='+ str(hp['seq_length_analysis'])+
              '\n'+str(hp['act_func'])+'+'+model_idx)


    plt.savefig(fig_path+'/'+'Evaluate_performance_'+str(hp['seq_length_analysis'])+'.pdf')
    plt.show()
#Evaluate_performance()

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

def neuron_activity(hp,trajectory_generator,model,fig_path):
    figure_path = os.path.join(fig_path, 'neuron_activity/'+str(hp['seq_length_analysis']))
    mkdir_p(figure_path)

    fs=10
    ########################################### speed =0 ######################################################
    inputs_0, pos_batch_0, _ = trajectory_generator.get_batch_for_test(speed_scale=0,batch_size=hp['batch_size_test'])
    firing_rate_binder_0 = model.grid_hidden(inputs_0)
    firing_rate_binder_0 = firing_rate_binder_0.reshape(firing_rate_binder_0.shape[1],firing_rate_binder_0.shape[0],firing_rate_binder_0.shape[2])
    print('firing_rate_binder_0',firing_rate_binder_0.shape)
    firing_rate_binder_0 = firing_rate_binder_0.detach().cpu().numpy()#.reshape(-1, hp['Ng'])

    ########################################### speed =1 ######################################################
    inputs, pos_batch, _ = trajectory_generator.get_batch_for_test(speed_scale=1,batch_size=hp['batch_size_test'])
    firing_rate_binder_1 = model.grid_hidden(inputs)
    firing_rate_binder_1 = firing_rate_binder_1.reshape(firing_rate_binder_1.shape[1],firing_rate_binder_1.shape[0],firing_rate_binder_1.shape[2])
    firing_rate_binder_1 = firing_rate_binder_1.detach().cpu().numpy().reshape(-1, hp['Ng'])
    print('firing_rate_binder_1',firing_rate_binder_1.shape)

    f = np.array(list([firing_rate_binder_1[i:(i+hp['batch_size_test']),:] for i in range(10) ]))
    print(f.shape)


    xstick = range(0,2400,20)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    fs = 15

    for unit_idx in range(0,2,1):
        plt.plot(f[:,:, unit_idx])
        plt.ylabel('Firing rate (a.u.)')
        plt.xlabel('Time (ms)')
        plt.title('unit_'+str(unit_idx))
        plt.tick_params(axis='both')
        plt.ylim([-0.05,0.12])
        plt.savefig(figure_path+'/'+ str(unit_idx)+'.png')
        plt.show()

    #plt.legend(loc="left",fontsize = 9)


def Calculate_percent_grid_ext_inh(hp,trajectory_generator,model,fig_path):
    figure_path = os.path.join(fig_path, 'Calculate_percent_grid_ext_inh/'+'score_60')
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

    #"""
    activations, _, _, _ = visualize.compute_ratemaps(model, trajectory_generator, hp,
                                                      res=res,
                                                      Ng=hp['Ng'],
                                                      n_avg=n_avg)


    ###################Compute a set of low-res maps to use for evalutaing grid score ###################
    _, rate_map_lores, _, _ = visualize.compute_ratemaps(model, trajectory_generator, hp,
                                                         res=low_res,
                                                         Ng=hp['Ng'],
                                                         n_avg=n_avg)
    score_60, score_90, sac, max_60_ind = zip(*[scorer.get_scores(rm.reshape(low_res, low_res)) for rm in tqdm(rate_map_lores)])

    score_type = score_60
    io.savemat(figure_path+'/'+"score_ext.mat",{'score':score_type})
    #"""
    ######################################################################################################

    load_data = io.loadmat(figure_path+'/'+"score_ext.mat")
    score_type = load_data['score'][0,:]

    score_exc = score_type[:410]
    idxs_exc = np.flip(np.argsort(score_exc))
    score_inh = score_type[410:]
    idxs_inh = np.flip(np.argsort(score_inh))


    k_exc=-1
    for i in idxs_exc:
        k_exc+=1
        if np.abs(score_exc[i])<0.3:
            break
    idx_exc = idxs_exc[:k_exc]
    k_inh=-1

    for j in idxs_inh:
        k_inh+=1
        if np.abs(score_inh[j])<0.3:
            break
    idx_inh = idxs_inh[:k_inh]

    percentage_E_grid = np.round(idx_exc.shape[0]/410,6)
    percentage_I_grid = np.round(idx_inh.shape[0]/102,6)
    print('percentage of E-grid:',percentage_E_grid)
    print('percentage of I-grid:',percentage_I_grid)


    # Plot high grid scores of inhibitory
    #"""

    n_plot = 25#number_score_high#128
    fig0 = plt.figure(figsize=(10,10))
    ax = fig0.add_axes([0.05, 0.05, 0.8, 0.8])
    rm_fig = visualize.plot_ratemaps_panel(activations[idx_exc], n_plot, smooth=True,width=5)
    plt.imshow(rm_fig)
    plt.title('percentage of E-grid: '+str(idx_exc.shape[0]/410),fontsize=16)
    plt.axis('off')
    plt.savefig(figure_path+'/high_grid_scores_E.png')
    #plt.show()

    ######################################inhibitory unit grid scale ###############################
    n_plot = 25#number_score_inh_high#100
    fig1 = plt.figure(figsize=(10,10))
    ax = fig1.add_axes([0.05, 0.05, 0.9, 0.9])

    rm_fig = visualize.plot_ratemaps_panel(activations[idx_inh+410], n_plot, smooth=True,width=5)
    plt.imshow(rm_fig)
    plt.suptitle('percentage of I-grid: '+str(idx_inh.shape[0]/102),fontsize=16)
    plt.axis('off')
    plt.savefig(figure_path+'/fig1B_high_score_I.png')
    #plt.show()
    #"""

    return percentage_E_grid, percentage_I_grid



def Plot_high_score_EI_panel_rgb(hp,trajectory_generator,model,fig_path):
    figure_path = os.path.join(fig_path, 'Plot_high_score_grid_EI_panel_rgb/'+'score_60')
    mkdir_p(figure_path)
    res = 50
    n_avg = 1
    #"""
    activations, _, _, _ = visualize.compute_ratemaps(model, trajectory_generator, hp,
                                                      res=res,
                                                      Ng=hp['Ng'],
                                                      n_avg=n_avg)

    ######################################################################################################
    data_path = os.path.join(fig_path, 'Calculate_percent_grid_ext_inh/'+'score_60')
    load_data = io.loadmat(data_path+'/'+"score_ext.mat")
    score_type = load_data['score'][0,:]

    score_exc = score_type[:410]
    idxs_exc = np.flip(np.argsort(score_exc))
    score_inh = score_type[410:]
    idxs_inh = np.flip(np.argsort(score_inh))
    ######################################################################################################

    n_plot = 100#number_score_high#128
    im = activations[idxs_exc,:,:]
    images = visualize.plot_ratemaps_panel_me(activations=activations, smooth=True)

    fig0 = plt.figure(figsize=(8,8))
    ax = fig0.add_axes([0.05, 0.05, 0.9, 0.9])
    j=0
    for idx in idxs_exc[:64]:
        j+=1
        plt.subplot(8,8,j)
        im = activations[idx,:,:]
        im=visualize.rgb(im)
        plt.imshow(im[:,:,0], interpolation='none', cmap='jet');plt.axis('off')
        if j==4:
            plt.title('high_scores_E'+'_sl_'+str(hp['seq_length_analysis']),fontsize=20);plt.axis('off')
    plt.savefig(figure_path+'/high_scores_E_rgb.png')
    plt.show()

    ######################################inhibitory unit grid scale ###############################
    fig0 = plt.figure(figsize=(8,8))
    ax = fig0.add_axes([0.05, 0.05, 0.9, 0.9])
    j=0
    for idx in idxs_inh[:64]:
        j+=1
        plt.subplot(8,8,j)
        im = activations[idx,:,:]
        im=visualize.rgb(im)
        plt.imshow(im[:,:,0], interpolation='none', cmap='jet');plt.axis('off')
        if j==4:
            plt.title('high_scores_I'+'_sl_'+str(hp['seq_length_analysis']),fontsize=20);plt.axis('off')
    plt.savefig(figure_path+'/high_scores_I_rgb.png')
    plt.show()

    # #"""


def Plot_grid_score_example_unit(hp,trajectory_generator,model,unit_idx,fig_path):
    figure_path = os.path.join(fig_path, 'Plot_grid_score_example_unit/'+'score_60')
    mkdir_p(figure_path)
    res = 50
    n_avg = 1

    low_res = 30
    starts = [0.2] * 10
    ends = np.linspace(0.4, 0.8, num=10)#np.linspace(0.4, 1.0, num=10)
    box_width=hp['box_width']
    box_height=hp['box_height']
    coord_range=((-box_width/2, box_width/2), (-box_height/2, box_height/2))
    masks_parameters = zip(starts, ends.tolist())
    scorer = GridScorer(low_res, coord_range, masks_parameters)



    activations, _, _, _ = visualize.compute_ratemaps(model, trajectory_generator, hp,
                                                      res=res,
                                                      Ng=hp['Ng'],
                                                      n_avg=n_avg)

    data_path = os.path.join(fig_path, 'Plot_grid_score_high_unit/'+'score_60')
    load_score = io.loadmat(data_path+'/'+"score_ext.mat")
    load_sacs = io.loadmat(data_path+'/'+"sacs_ext.mat")
    score_type = load_score['score'][0,:]
    sacs = load_sacs['sac']

    for idx in unit_idx:

        fig,axs = plt.subplots(1,2,figsize=(6,3))
        im = activations[idx,:,:]
        im = (im - np.min(im)) / (np.max(im) - np.min(im))# normalization
        im = cv2.GaussianBlur(im, (3,3), sigmaX=1, sigmaY=0)
        axs[0].imshow(im, interpolation='none', cmap='jet');axs[0].axis('off')

        im_correlation = sacs[idx,:,:]
        scorer.plot_sac(im_correlation,ax=axs[1])
        #image_correlation = visualize.rgb(im_correlation, smooth=True)
        fig.suptitle('unit_'+str(idx)+'; score:' +str(np.round(score_type[idx],2)))
        plt.savefig(figure_path+'/'+'unit_'+str(idx)+'_'+str(np.round(score_type[idx],2))+'.png')
        plt.show()


    #"""





def Plot_ratemap_manipulate_weight(get_grid,hp,scale,data_path):
    hp['get_grid'] =get_grid
    # hp['get_grid']='EC'
    # hp['get_grid'] = 'DG'
    # hp['get_grid'] = 'CA3'

    # hp['get_grid'] = 'CA1'
    #hp['get_grid'] = 'decoder'

    if get_grid=='EC':
        idx_select = [63,74,97]

    if get_grid=='CA3':
        idx_select = [91,11,40,55]


    if get_grid == 'DG':
        idx_select = [91,150,223,251]


    if get_grid == 'CA1':
        idx_select = [3,71,223,23]


    # res = 50
    # n_avg = 2
    # activations, _, _, _ = visualize.compute_ratemaps(model, trajectory_generator, hp,
    #                                                   res=res,
    #                                                   Ng=hp['Ng'],
    #                                                   n_avg=n_avg)
    #
    # np.save(data_path+get_grid+'_'+hp['env']+'_activations.npy',activations)

    activations = np.load(data_path+get_grid+'_'+hp['env']+'_activations.npy')









    #### each unit###########

    for idx in np.array(idx_select):
        # print('=============idx',idx)
        fig1 = plt.figure(figsize=(2,2))
        ax = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
        # plt.title(hp['get_grid'] + '_' + str(idx)+'; '+str(hp['degree'])+'_'+str(scale), fontsize=8)


        # print(get_grid, 'activations',activations[idx])
        im = activations[idx,:,:]
        im = (im - np.min(im)) / (np.max(im) - np.min(im))  # normalization
        # im = cv2.GaussianBlur(im, (3, 3), sigmaX=1, sigmaY=0)

        # Define the circular region
        grid_size = im.shape[0]
        x = np.linspace(-hp['box_width']/2, hp['box_width']/2, grid_size)
        y = np.linspace(-hp['box_width']/2, hp['box_width']/2, grid_size)
        xx, yy = np.meshgrid(x, y)
        distance_from_center = np.sqrt(xx ** 2 + yy ** 2)

        # Mask for the circular region
        circle_radius = hp['box_width']/2  # Radius of the circle
        circular_mask = distance_from_center <= circle_radius

        # Apply the mask to the activations
        masked_im = np.full_like(im, np.nan)  # Fill with NaN outside the circle
        masked_im[circular_mask] = im[circular_mask]

        ax.imshow(masked_im, interpolation='gaussian', cmap='jet');ax.axis('off')
        # plt.savefig(figure_path+'/'+str(idx)+hp['get_grid']+'_'+str(hp['degree'])+'_'+str(scale)+hp['env']+'.pdf')
        plt.show()

def Plot_ratemap_manipulate_weight_rectangle(get_grid,hp,scale,data_path):
    hp['get_grid'] =get_grid
    # hp['get_grid']='EC'
    # hp['get_grid'] = 'DG'
    # hp['get_grid'] = 'CA3'

    # hp['get_grid'] = 'CA1'
    #hp['get_grid'] = 'decoder'
    if get_grid == 'EC':
        idx_select = [63, 74, 97]

    if get_grid == 'CA3':
        idx_select = [91,11,40,55]

    if get_grid == 'DG':
        idx_select = [ 91, 150, 223, 251]

    if get_grid == 'CA1':
        idx_select = [3,71,223,23]


    # res = 50
    # n_avg = 2
    # activations, _, _, _ = visualize.compute_ratemaps(model, trajectory_generator, hp,
    #                                                   res=res,
    #                                                   Ng=hp['Ng'],
    #                                                   n_avg=n_avg)
    # np.save(data_path+get_grid+'_'+hp['env']+'_activations.npy',activations)
    #



    activations = np.load(data_path + get_grid + '_' + hp['env'] + '_activations.npy')



    ######Compute a set of low-res maps to use for evalutaing grid score
    #### each unit###########
    for idx in np.array(idx_select):
        fig1 = plt.figure(figsize=(2, 2))
        ax = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
        # plt.title(hp['get_grid'] + '_' + str(idx)+'; '+str(hp['degree'])+'_'+str(scale), fontsize=15)

        # print(get_grid, 'activations', activations[idx])
        im = activations[idx, :, :]
        im = (im - np.min(im)) / (np.max(im) - np.min(im))  # normalization
        # im = cv2.GaussianBlur(im, (3, 3), sigmaX=1, sigmaY=0)
        ax.imshow(im, interpolation='gaussian', cmap='jet');
        ax.axis('off')

        # plt.savefig(figure_path+'/'+str(idx)+hp['get_grid']+'_'+str(hp['degree'])+'_'+str(scale)+hp['env']+'.pdf')
        plt.show()








if __name__ == '__main__':
    pass



    # for unit_idx in range(5):
    #     visual_tuning1(unit_idx)


