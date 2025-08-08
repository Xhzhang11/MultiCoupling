import sys,os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ranksums
from scipy.stats import wilcoxon


from scipy.io import savemat

import seaborn as sns


hp={}

hp['rng']=np.random.RandomState(100)
######

#=========================  plot =============================

hp['root_path'] = os.path.abspath(os.path.join(os.getcwd(),"./"))


data_root = hp['root_path']+'/Datas/'
data_path = os.path.join(data_root, 'angle_pca/')

print('data_path',data_path)




def get_model(model_idx,idx):
    hp['n_input'] = hp['n_eachring'] * 2 + hp['n_cue'] + hp['n_rule']
    hp['n_output'] = hp['n_eachring']


    model_prefix = str(hp['activation']) + '_' + str(hp['type']) + '_' + str(hp['n_rnn']) + '_bs' + str(
        hp['batch_size']) + '_stim' + str(hp['stim']) + '_pc' + str(hp['sparsity_cross']) + '_sp' + str(hp['sparsity_md2ppc']) + '_sf' + str(
        hp['sparsity_md2pfc']) + '_ai' + str(hp['alpha_input']) + '_ao' + str(hp['alpha_output']) + '_'


    model_name = model_prefix + str(model_idx)

    local_folder_name = os.path.join('/' +'model_'+ hp['rule_model'], model_name, str(idx))
    #local_folder_name = os.path.join('/' + 'model_both', model_name)
    model_dir = hp['root_path'] + local_folder_name + '/'

    print('model_dir',model_dir)


    if os.path.exists(model_dir):
        print(f"The path '{model_dir}' exists.")
    else:
        print(f"The path '{model_dir}' does not exist.")
        sys.exit(0)

    return model_dir,model_name



def get_model_hp():
    hp['n_eachring'] = 64
    hp['n_angle'] = 8
    hp['cuestim'] = hp['stim']
    hp['stim_delay'] = hp['stim']
    hp['cue_delay'] = hp['stim']
    hp['resp'] = 40

    hp['rule_model'] = 'both'
    hp['cost_strength'] = 5.0
    hp['activation'] = 'softplus'  # softplus,relu
    hp['dt'] = 10

    hp['n_md'] = 64
    hp['input_scale'] = 1.0
    hp['tau_ppc'] = 40
    hp['tau_pfc'] = 20
    hp['batch_size'] = 512
    hp['learning_rate'] = 0.0001




def calculate_angle(model_idx,idx,N_random):
    get_model_hp()

    hp['type'] = 'type1'
    hp['activation'] = 'softplus'

    hp['sparsity_cross'] = 0.8
    hp['sparsity_md2ppc'] = 0.5
    hp['sparsity_md2pfc'] = 0.5

    hp['alpha_input'] = 0.3
    hp['alpha_output'] = 0.5
    hp['n_rnn'] = 200
    hp['stim'] = 100
    hp['in_strength'] = 0.5


    model_dir, model_name = get_model(model_idx, idx)
    epoch = angle_calculate_lib.get_epoch(model_dir, rule_name='retro', hp=hp)
    fig_path_0 = os.path.join(figure_path, model_name)
    tools.mkdir_p(fig_path_0)

    print('model_name',model_name)

    hp['plot_activtity'] = True#False#


    hp['seed'] = 0



    # angle_calculate_lib.PCA_plot_3D_selected_angle_pfc_paper(fig_path_0, model_name, model_dir, idx, hp,rule_name='retro',
    #                                                                start_proj=epoch['stim1_off'][0],
    #                                                                end_proj=epoch['stim2_off'][0])

    angle_calculate_lib.PCA_h1h2(fig_path_0, model_name, model_dir, idx, hp, rule_name='retro',
                                                   start_proj=epoch['stim1_off'][0],
                                                   end_proj=epoch['stim2_off'][0])
# calculate_angle(model_idx=0,idx=15,N_random=0)





def calculate_angle_all(model_idx,idx,N_random):
    get_model_hp()

    hp['type'] = 'type1'
    hp['activation'] = 'softplus'

    hp['sparsity_cross'] = 0.8
    hp['sparsity_md2ppc'] = 0.5
    hp['sparsity_md2pfc'] = 0.5

    hp['alpha_input'] = 0.3
    hp['alpha_output'] = 0.5
    hp['n_rnn'] = 200
    hp['stim'] = 100
    hp['in_strength'] = 0.5


    model_dir, model_name = get_model(model_idx, idx)
    epoch = angle_calculate_lib.get_epoch(model_dir, rule_name='retro', hp=hp)



    print('model_name',model_name)

    hp['plot_activtity'] = True#False#




    angle_calculate_lib.PCA_plot_3D_selected_angle_pfc(figure_path, data_path,model_name, model_dir, idx, hp,rule_name='retro',
                                                                   start_proj=epoch['stim1_off'][0],
                                                                   end_proj=epoch['stim2_off'][0])
#
# for model_idx in range(0,60):#4,5,6,7,8
#     for idx in np.array([15]):#0,2,4,6,8,10,12,14,16,20,22
#         for seed in range(5):
#             hp['seed'] = seed
#             calculate_angle_all(model_idx,idx,N_random=0)




def calculate_angle_PPC_PFC(model_idx,idx,control_scale,rule_name):
    get_model_hp()

    hp['type'] = 'type1'
    hp['activation'] = 'softplus'

    hp['sparsity_cross'] = 0.8
    hp['sparsity_md2ppc'] = 0.5
    hp['sparsity_md2pfc'] = 0.5

    hp['alpha_input'] = 0.3
    hp['alpha_output'] = 0.5
    hp['n_rnn'] = 200
    hp['stim'] = 100
    hp['control_scale']=control_scale
    hp['in_strength'] = 1.0


    model_dir, model_name = get_model(model_idx, idx)
    epoch = angle_calculate_lib.get_epoch(model_dir, rule_name='retro', hp=hp)




    print('=========================model_name',model_name)

    hp['plot_activtity'] = True#False#
    angles1=0;angles2=0

    angles1 = angle_calculate_lib.PCA_plot_angle_PPC_PFC(figure_path, data_path,model_name, model_dir, idx, hp,
                                                 rule_name=rule_name,
                                               start_proj=epoch['stim1_off'][0]+2,
                                               end_proj=epoch['stim2_on'][0]-2)

    angles2 = angle_calculate_lib.PCA_plot_angle_PPC_PFC(figure_path, data_path, model_name, model_dir, idx, hp,
                                               rule_name=rule_name,###
                                               start_proj=epoch['stim2_off'][0]+2,
                                               end_proj=epoch['response_on'][0]-2)

    success_action_prob = angle_calculate_lib.get_performance(model_dir,hp)

    print('====angles1,angles2',angles1,angles2)
    return angles1,angles2,success_action_prob


def get_data_angle_PPC_PFC(data_path_0,rule_name,model_idx,idx,scale_values):

    angles1_p0_list = []
    angles1_p1_list = []
    angles2_p0_list = []
    angles2_p1_list = []
    success_action_prob_list = []

    for control_scale in np.array(scale_values):#[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.1]
        print('control_scale=========================================', control_scale)
        angles1_p0_temp = []
        angles1_p1_temp = []
        angles2_p0_temp = []
        angles2_p1_temp = []
        success_action_prob_temp = []

        for seed in range(20):
            hp['seed'] = seed
            angles1,angles2,success_action_prob = calculate_angle_PPC_PFC(model_idx=model_idx,idx=idx,
                                                                          control_scale=control_scale,rule_name=rule_name)

            angles1_p0_temp.append(angles1[0])
            angles1_p1_temp.append(angles1[1])
            angles2_p0_temp.append(angles2[0])
            angles2_p1_temp.append(angles2[1])

            success_action_prob_temp.append(success_action_prob)

        angles1_p0_list.append(angles1_p0_temp)
        angles1_p1_list.append(angles1_p1_temp)
        angles2_p0_list.append(angles2_p0_temp)
        angles2_p1_list.append(angles2_p1_temp)

        success_action_prob_list.append(success_action_prob_temp)


        np.save(data_path_0 + 'angles1_p0_list.npy', angles1_p0_list)
        np.save(data_path_0 + 'angles1_p1_list.npy', angles1_p1_list)
        np.save(data_path_0 + 'angles2_p0_list.npy', angles2_p0_list)
        np.save(data_path_0 + 'angles2_p1_list.npy', angles2_p1_list)
        np.save(data_path_0+ 'success_action_prob_list.npy', success_action_prob_list)




def plot_angle_PPC_PFC(rule_name):
    model_idx=50
    idx=15
    data_path_0 = os.path.join(data_path,'angle_'+str(model_idx)+'_'+str(idx) + rule_name + '/')

    scale_values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.1, 0.01]


    # get_data_angle_PPC_PFC(data_path_0,rule_name,model_idx,idx,scale_values)

    angles1_p0_list = np.load(data_path_0 + 'angles1_p0_list.npy')
    angles1_p1_list = np.load(data_path_0 + 'angles1_p1_list.npy')
    angles2_p0_list = np.load(data_path_0 + 'angles2_p0_list.npy')
    angles2_p1_list = np.load(data_path_0 + 'angles2_p1_list.npy')


    print('angles1_p0_list',angles1_p0_list.shape,angles1_p0_list)
    print('angles1_p1_list', angles1_p1_list)


    value_idx = [0,1,3,8]

    delay1_p0_list_baseline = angles1_p0_list[0,:]
    delay1_p1_list_baseline = angles1_p1_list[0,:]
    delay2_p0_list_baseline = angles2_p0_list[0, :]
    delay2_p1_list_baseline = angles2_p1_list[0, :]

    delay1_p0_list_pert1 = angles1_p0_list[1,:]
    delay1_p1_list_pert1 = angles1_p1_list[1,:]
    delay2_p0_list_pert1 = angles2_p0_list[1, :]
    delay2_p1_list_pert1 = angles2_p1_list[1, :]

    delay1_p0_list_pert2 = angles1_p0_list[3, :]
    delay1_p1_list_pert2 = angles1_p1_list[3, :]
    delay2_p0_list_pert2 = angles2_p0_list[3, :]
    delay2_p1_list_pert2 = angles2_p1_list[3, :]

    delay1_p0_list_pert3 = angles1_p0_list[8, :]
    delay1_p1_list_pert3 = angles1_p1_list[8, :]
    delay2_p0_list_pert3 = angles2_p0_list[8, :]
    delay2_p1_list_pert3 = angles2_p1_list[8, :]



    print('delay2_p1_list_pert3',delay2_p1_list_pert3.shape)






    data_baseline_0 = np.array([delay1_p0_list_baseline, delay1_p1_list_baseline,
                              delay2_p0_list_baseline, delay2_p1_list_baseline
                              ]).T


    data_pert_1 = np.array([delay1_p0_list_pert1, delay1_p1_list_pert1,
                              delay2_p0_list_pert1, delay2_p1_list_pert1]).T

    data_pert_2 = np.array([delay1_p0_list_pert2, delay1_p1_list_pert2,
                            delay2_p0_list_pert2, delay2_p1_list_pert2]).T

    data_pert_3 = np.array([delay1_p0_list_pert3, delay1_p1_list_pert3,
                            delay2_p0_list_pert3, delay2_p1_list_pert3]).T







    colors_list = sns.color_palette("Set2")
    colors_0 = [colors_list[0], colors_list[0], colors_list[0], colors_list[0], ]
    colors_1 = [colors_list[1], colors_list[1], colors_list[1], colors_list[1], ]
    colors_2 = [colors_list[2], colors_list[2], colors_list[2], colors_list[2], ]
    colors_3 = [colors_list[3], colors_list[3], colors_list[3], colors_list[3], ]

    width=0.7
    # fig = plt.figure(figsize=(2.5, 2.5))
    # ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    fig = plt.figure(figsize=(2, 2.5))
    ax = fig.add_axes([0.3, 0.2, 0.6, 0.7])
    plt.title(rule_name+':baseline')
    # plt.boxplot(data)
    sns.violinplot(data=data_baseline_0, palette=colors_0, linewidth=1,width=width,linecolor="k", inner='point', alpha=1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel('', fontsize=10)
    plt.ylim([0, 92])
    # plt.savefig(figure_path + 'angle_' + rule_name + '_baseline.pdf')
    plt.show()

    fig = plt.figure(figsize=(2, 2.5))
    ax = fig.add_axes([0.3, 0.2, 0.6, 0.7])
    plt.title(rule_name+':perturbation1')
    sns.violinplot(data=data_pert_1, palette=colors_1,  linewidth=1,width=width,linecolor="k",inner='point', alpha=1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel('', fontsize=10)
    plt.ylim([0, 92])
    # plt.savefig(figure_path + 'angle_' + rule_name + '_perturb1.pdf')
    plt.show()

    fig = plt.figure(figsize=(2, 2.5))
    ax = fig.add_axes([0.3, 0.2, 0.6, 0.7])
    plt.title(rule_name + ':perturbation2')
    sns.violinplot(data=data_pert_2, palette=colors_2, linewidth=1,width=width,linecolor="k",inner='point', alpha=1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel('', fontsize=10)
    plt.ylim([0, 92])
    # plt.savefig(figure_path + 'angle_' + rule_name + '_perturb2.pdf')
    plt.show()


    #
    #
    #
    #
    #
    fig = plt.figure(figsize=(2, 2.5))
    ax = fig.add_axes([0.3, 0.2, 0.6, 0.7])
    plt.title(rule_name+':perturbation3')
    sns.violinplot(data=data_pert_3, palette=colors_3, linewidth=1,width=width,linecolor="k",inner='point', alpha=1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel('', fontsize=10)
    plt.ylim([0, 92])
    # plt.savefig(figure_path + 'angle_' + rule_name+'_perturb3.pdf')
    plt.show()

    # print('angles1_p0_list',angles1_p0_list)


plot_angle_PPC_PFC(rule_name='retro')
plot_angle_PPC_PFC(rule_name='prosp')








def plot_perf_PPC_PFC(rule_name):
    model_idx = 50
    idx = 15
    data_path_0 = os.path.join(data_path, 'angle_' + str(model_idx) + '_' + str(idx) + rule_name + '/')


    scale_values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.1, 0.01]#scale_perturb = [1.0,0.9,0.7,0.01]



    # get_data_angle_PPC_PFC(data_path_0,rule_name,model_idx,idx,scale_values)

    success_action_prob_list = np.load(data_path_0 + 'success_action_prob_list.npy')
    print('success_action_prob_list',success_action_prob_list.shape)
    success_action_prob_0 = success_action_prob_list[0, :]
    success_action_prob_1 = success_action_prob_list[1, :]
    success_action_prob_2 = success_action_prob_list[3, :]
    success_action_prob_3 = success_action_prob_list[8, :]


    mean_0 = np.mean(success_action_prob_0, axis=0)
    mean_1 = np.mean(success_action_prob_1, axis=0)
    mean_2 = np.mean(success_action_prob_2, axis=0)
    mean_3 = np.mean(success_action_prob_3, axis=0)

    error_0 = np.std(success_action_prob_0, axis=0)
    error_1 = np.std(success_action_prob_1, axis=0)
    error_2 = np.std(success_action_prob_2, axis=0)
    error_3 = np.std(success_action_prob_3, axis=0)

    mean_success_action_prob_list = [mean_0,mean_1,mean_2,mean_3]
    error_success_action_prob_list = [error_0, error_1, error_2, error_3]



    scale_idx = [0,1,2,3]

    colors_list = sns.color_palette("Set2")

    fig = plt.figure(figsize=(2.2, 2.2))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    plt.plot(scale_idx,mean_success_action_prob_list, linewidth=1,color='k')


    # Plot error bars and colored markers one by one
    for i in range(len(scale_idx)):
        ax.errorbar(scale_idx[i], mean_success_action_prob_list[i], yerr=error_success_action_prob_list[i],
                    fmt='o', color=colors_list[i], ecolor=colors_list[i], capsize=2, markersize=5)

    # plt.errorbar(scale_idx, mean_success_action_prob_list,error_success_action_prob_list, markersize=5,color='k', fmt='o')

    plt.title(rule_name, fontsize=10)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks([0,1,2,3])
    # plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=12)
    plt.ylabel('Performance', fontsize=12)
    # plt.savefig(figure_path + 'performance_' + rule_name+'.pdf')


    plt.show()


plot_perf_PPC_PFC(rule_name='retro')
plot_perf_PPC_PFC(rule_name='prosp')
#























