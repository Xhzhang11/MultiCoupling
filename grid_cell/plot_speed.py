
import sys, os
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"../..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"../")))

from matplotlib import pyplot as plt
import seaborn as sns
import json


hp = {}
hp['root_path'] = os.path.abspath(os.path.join(os.getcwd(),"./"))



hp['learning_rate']=0.0001
hp['batch_size_test']=5000

hp['rotate'] = True
hp['sparse_EC_CA1']=0
hp['sparse_CA3_CA1']=0
hp['is_EI'] = 'EIno'
hp['mode'] = 'test'
# hp['degree'] = 90#0,30,45,60,90

hp['save_dir'] = hp['root_path']+'/model'
print('save_dir', hp['save_dir'])


fig_path = hp['root_path'] + '/Figures/'
figure_path = os.path.join(fig_path, 'plot_loss'+'/')



data_root = hp['root_path'] + '/Datas/'
data_path = os.path.join(data_root, 'gridcell_dffinput_2.20_0/')




def get_model_dir_diff_context(model_idx):
    hp['run_ID'] = hp['type']+'_'+hp['activation']+'_'+hp['activation_CA3']\
				   + '_wDG' + str(hp['wDG'])+ '_wc' + str(hp['wc'])+ '_wa' + str(hp['wa'])\
				   +'_sp'+str(hp['sparsity'])+'_'+ str(model_idx)

    local_folder_name = os.path.join('/' + 'model', hp['run_ID'])
    model_dir = hp['root_path'] + local_folder_name#+'/'
    #print(model_dir)
    #
    # if os.path.exists(model_dir):
    #     print(f"The path '{model_dir}' exists.")
    #     #print("The path exists.")
    # else:
    #     print(f"The path '{model_dir}' does not exist.")
    #     sys.exit(0)
    run_ID = hp['run_ID']
    return model_dir,run_ID

def plot_error_curve(wDG,wc,wa,scale,env):
    # hp['env'] = 'rectangle'
    # hp['env'] = 'circle'

    hp['env'] = env


    hp['CA1_constrain'] = 'yes'
    hp['activation'] = 'relu'
    hp['activation_CA3']='sigmoid'
    hp['activation_md']='relu'


    hp['type'] = 'type1'
    hp['freeze'] = 'no'
    hp['sparsity'] = 0.5
    hp['init']='rand'
    hp['wDG'] = wDG
    hp['wc'] = wc
    hp['wa'] = wa
    hp['scale_CA3_CA1'] = 1
    hp['scale_EC_CA1'] = 1
    # hp['scale_vis']=100

    # hp['type'] = 'type1'
    # model_idx = 0
    # model_dir,run_ID = get_model_dir_diff_context(model_idx=model_idx)
    # error_data_type1 = np.load(model_dir + '/err_list.npy')
    # print('model_dir',model_dir)
    #
    #
    # ##
    # hp['type'] = 'type0'
    # model_idx = 3
    # model_dir, run_ID = get_model_dir_diff_context(model_idx=model_idx)
    # error_data_type0 = np.load(model_dir + '/err_list.npy')
    # print('model_dir', model_dir)
    #
    # np.save(data_path + 'error_data_type1.npy', error_data_type1)
    # np.save(data_path + 'error_data_type0.npy', error_data_type0)





    error_data_type1 = np.load(data_path + 'error_data_type1.npy')
    error_data_type0 = np.load(data_path + 'error_data_type0.npy')









    font_ticket = 12

    fig = plt.figure(figsize=(3, 2.5))
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.75])

    plt.plot(error_data_type1,c='tab:orange', label='Multi.')
    plt.plot(error_data_type0,c='tab:blue', label='Add.')
    plt.title('decoding error', fontsize=font_ticket)
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    plt.xticks([0,300000,600000], fontsize=font_ticket)
    plt.yticks([0.2, 0.4, 0.6, 0.8,1.0], fontsize=font_ticket)
    plt.xlabel('train step', fontsize=font_ticket)
    plt.ylabel('error', fontsize=font_ticket)
    plt.legend(fontsize=10)
    # fig.savefig(figure_path + 'plot_error' + '.pdf')
    plt.show()



    # fig = plt.figure(figsize=(4, 3.5))
    # ax = fig.add_axes([0.15, 0.15, 0.8, 0.75])
    # ax.plot(error_data, label='Error')  # Plot with a label
    # ax.set_xlabel('Epochs')  # X-axis label
    # ax.set_ylabel('Error')  # Y-axis label
    # ax.set_title('Training Error (Log Scale)')  # Title
    # ax.set_xscale('log')  # Logarithmic y-axis
    # ax.set_yscale('log')  # Logarithmic y-axis
    # ax.legend()  # Legend
    # plt.show()

plot_error_curve(wDG=0.4, wc=1.0, wa=0.3, scale=1.0, env='circle')  ####




def plot_violinplot():


    # # scale_list = []
    # epoch_0_list = []
    # epoch_1_list = []
    #
    #
    # for model_idx in range(15):
    #     epoch0 = plot_speed(model_idx=model_idx, type='type0')
    #     epoch_0_list.append(epoch0)
    #
    # for model_idx in range(15):
    #     epoch1 = plot_speed(model_idx=model_idx, type='type1')
    #     epoch_1_list.append(epoch1)
    #
    #
    # np.save(data_path + 'epoch_0_list.npy', epoch_0_list)
    # np.save(data_path + 'epoch_1_list.npy', epoch_1_list)

    epoch_0_list = list(np.load(data_path + 'epoch_0_list.npy'))
    epoch_1_list = list(np.load(data_path + 'epoch_1_list.npy'))





    print( '==epoch_0',len(epoch_0_list),epoch_0_list)
    print( '==epoch_1',len(epoch_1_list),epoch_1_list)
    data_violin = np.array([epoch_0_list, epoch_1_list]).T

    IT_0 = np.mean(epoch_0_list)
    IT_1 = np.mean(epoch_1_list)

    IT_std_0 = np.std(epoch_0_list) / np.sqrt(len(epoch_0_list))
    IT_std_1 = np.std(epoch_1_list) / np.sqrt(len(epoch_1_list))


    IT_mean = [IT_0, IT_1]
    IT_std = [IT_std_0, IT_std_1]

    from scipy import stats
    p01 = stats.ranksums(epoch_0_list, epoch_1_list)


    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_axes([0.3, 0.1, 0.55, 0.7])

    colors = sns.color_palette()  # sns.color_palette("Set2")
    sns.violinplot(data=data_violin, palette=[colors[0],colors[1]], linecolor="k", linewidth=0.7,inner='point', alpha=0.9)


    plt.ylabel('trials', fontsize=12)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title('GridCell: RNN='+hp['activation']+'; FNN='+hp['activation_md']+'\n'+'p='+str(p01[1])+'\n', fontsize=8)
    # plt.xlim([-0.4, 1.5])
    # plt.yticks([0, 2, 4, 6], fontsize=13)
    #fig.savefig(figure_path  +'speed.png')
    # fig.savefig(figure_path +'speed_violinplot.pdf')
    plt.show()


plot_violinplot()






