import sys,os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ranksums
from scipy.stats import wilcoxon

import default
import tools
import run
import decode_lib_test
from scipy.io import savemat
from scipy.stats import sem


hp=default.get_default_hp(rule_name='prosp')

hp['rng']=np.random.RandomState(0)
############ model
import seaborn as sns


# define input and output


#=========================  plot =============================

hp['root_path'] = os.path.abspath(os.path.join(os.getcwd(), "./"))

fig_path = hp['root_path']+'/Figures/'
figure_path = os.path.join(fig_path, 'plot_decode_test/')
tools.mkdir_p(figure_path)
data_path = os.path.join(hp['root_path'], 'Datas','plot_decode_test/')
tools.mkdir_p(data_path)



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
    hp['resp'] =40



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


def plot_select_decode(model_idx, idx):
    get_model_hp()

    hp['type'] = 'type1'
    hp['activation'] = 'softplus'

    hp['sparsity_cross']=0.8
    hp['sparsity_md2ppc']=0.5
    hp['sparsity_md2pfc']=0.5

    hp['alpha_input']=0.3
    hp['alpha_output']=0.5
    hp['n_rnn'] = 200

    hp['stim'] = 100


    fig_path1 = os.path.join(figure_path, 'plot_select_decode' + 'paper/')
    tools.mkdir_p(fig_path1)
    model_dir, model_name = get_model(model_idx, idx)
    epoch = decode_lib_test.get_epoch(model_dir=model_dir, rule_name='retro', hp=hp)
    stim2_on = epoch['stim2_on'][0];
    stim2_off = epoch['stim2_off'][0]
    stim1_off = epoch['stim1_off'][0]
    suffix = tools.splid_model_name(model_name, start_str='ao')
    file_name = suffix + '_' + str(idx)

    # accuracy_ppc_list = []
    # accuracy_pfc_list = []
    # for seed in range(10):
    #     hp['seed'] = seed
    #
    #     accuracy_ppc,accuracy_pfc = decode_lib_test.decode_up_low_both_new(figure_path, model_name, model_dir, idx, hp, sigma_x=0.3)
    #
    #     accuracy_ppc_list.append(accuracy_ppc)
    #     accuracy_pfc_list.append(accuracy_pfc)
    #
    # accuracy_ppc_list=np.array(accuracy_ppc_list)
    # accuracy_pfc_list = np.array(accuracy_pfc_list)
    # np.save(data_path+'accuracy_ppc_list'+file_name+'selection_'+'.npy',accuracy_ppc_list)
    # np.save(data_path+'accuracy_pfc_list'+file_name+'selection_'+'.npy',accuracy_pfc_list)



    accuracy_ppc_list = np.load(data_path + 'accuracy_ppc_list' + file_name + 'selection_'  + '.npy')
    accuracy_pfc_list = np.load(data_path + 'accuracy_pfc_list' + file_name + 'selection_'  + '.npy')


    mean_ppc = np.mean(accuracy_ppc_list, axis=0)
    sem_ppc = sem(accuracy_ppc_list, axis=0)  # / np.sqrt(accuracy_ppc_list.shape[0])

    mean_pfc = np.mean(accuracy_pfc_list, axis=0)
    sem_pfc = sem(accuracy_pfc_list, axis=0)  # / np.sqrt(accuracy_pfc_list.shape[0])

    fig = plt.figure(figsize=(2.7, 2.7))
    ax = fig.add_axes([0.25, 0.2, 0.7, 0.6])
    plt.title(file_name + ': selection', fontsize=9)
    start = stim1_off + 1

    xs = np.arange(0, mean_ppc.shape[0], 1)
    plt.plot(xs, mean_ppc, c='tab:blue', label='PPC')
    plt.fill_between(xs, mean_ppc - sem_ppc, mean_ppc + sem_ppc, color='tab:blue', alpha=0.3)
    # axs.axvline(stim2_on-start, color='darkgrey', linestyle='--')
    ax.axvspan(stim2_on - start, stim2_off - start, color='lightgrey')
    ax.axhline(y=0.5, color='black', linewidth=1)
    plt.xlim([-1.1, 20 + 3])
    plt.xticks([-1, 4, 9, 14, 19])




    plt.ylim([0.4, 1.01])
    plt.plot(mean_pfc, c='tab:purple', label='PFC')
    plt.fill_between(xs, mean_pfc - sem_pfc, mean_pfc + sem_pfc, color='tab:purple', alpha=0.3)
    plt.xlabel('from Cue_on')
    plt.ylabel('classifier accuracy')
    ax.spines[['right', 'top']].set_visible(False)

    plt.legend(fontsize=8)
    plt.savefig(fig_path1 + file_name + '_' + str(idx) + '_' + str(hp['sigma_x']) + '_'  + 'selection.pdf')
    plt.show()


# for model_idx in np.array([50]):  # 3,13,23
#     for idx in np.array([15]):  # 23
#         plot_select_decode(model_idx, idx)


#


def plot_decode_generalize(model_idx,idx,generalize,control_scale):
    get_model_hp()



    hp['type'] = 'type1'
    hp['activation'] = 'softplus'

    hp['sparsity_cross']=0.8
    hp['sparsity_md2ppc']=0.5
    hp['sparsity_md2pfc']=0.5

    hp['alpha_input']=0.3
    hp['alpha_output']=0.5
    hp['n_rnn'] = 200

    hp['stim'] = 100
    hp['sigma_x']=0.1

    control_scales = [1,0.5,1.1]

    hp['control_scale'] = control_scale#1.0


    model_dir, model_name = get_model(model_idx, idx)
    epoch = decode_lib_test.get_epoch(model_dir=model_dir, rule_name='retro', hp=hp)
    stim2_on = epoch['stim2_on'][0];
    stim2_off = epoch['stim2_off'][0]
    stim1_off = epoch['stim1_off'][0]
    suffix = tools.splid_model_name(model_name, start_str='ao')
    file_name = suffix + '_' + str(idx)

    fig_path1 = os.path.join(figure_path, 'plot_decode_generalize_'+str(hp['sigma_x']) + 'paper/')
    tools.mkdir_p(fig_path1)


    #
    # accuracy_ppc_list = []
    # accuracy_pfc_list = []
    # for seed in range(30):#select some seed to plot
    #     print('seed',seed)
    #     hp['seed'] = seed
    #
    #     if generalize=='retro2prosp':
    #         accuracy_ppc,accuracy_pfc=decode_lib_test.decode_up_low_both_generalize_retro2prosp(figure_path, model_name, model_dir, idx, hp)
    #
    #     elif generalize=='prosp2retro':
    #         accuracy_ppc,accuracy_pfc=decode_lib_test.decode_up_low_both_generalize_prosp2retro(figure_path, model_name, model_dir, idx, hp)
    #
    #     if np.mean(accuracy_ppc[0:3])<0.65 and np.mean(accuracy_pfc[0:3])<0.65:
    #         accuracy_ppc_list.append(accuracy_ppc)
    #         accuracy_pfc_list.append(accuracy_pfc)
    #
    # accuracy_ppc_list=np.array(accuracy_ppc_list)
    # accuracy_pfc_list = np.array(accuracy_pfc_list)
    # np.save(data_path+'accuracy_ppc_list'+file_name+'_'+str(hp['control_scale'])+'_'+generalize+'.npy',accuracy_ppc_list)
    # np.save(data_path+'accuracy_pfc_list'+file_name+'_'+str(hp['control_scale'])+'_'+generalize+'.npy',accuracy_pfc_list)





    accuracy_ppc_list = np.load(data_path+'accuracy_ppc_list'+file_name+'_'+str(hp['control_scale'])+'_'+generalize+'.npy')
    accuracy_pfc_list = np.load(data_path + 'accuracy_pfc_list' + file_name+'_'+str(hp['control_scale']) + '_' + generalize + '.npy')


    print('accuracy_ppc_list,accuracy_pfc_list',accuracy_pfc_list.shape,accuracy_pfc_list.shape)

    accuracy_ppc_list = accuracy_ppc_list[0:10]
    accuracy_pfc_list = accuracy_pfc_list[0:10]





    mean_ppc = np.mean(accuracy_ppc_list, axis=0)
    sem_ppc = sem(accuracy_ppc_list, axis=0)# / np.sqrt(accuracy_ppc_list.shape[0])

    mean_pfc = np.mean(accuracy_pfc_list, axis=0)
    sem_pfc = sem(accuracy_pfc_list, axis=0)# / np.sqrt(accuracy_pfc_list.shape[0])

    fig = plt.figure(figsize=(2.7, 2.7))
    ax = fig.add_axes([0.25, 0.2, 0.7, 0.6])
    plt.title(file_name+'_scale'+str(hp['control_scale']) +': '+generalize, fontsize=9)
    start = stim1_off + 1

    xs = np.arange(0, mean_ppc.shape[0], 1)
    plt.plot(xs, mean_ppc, c='tab:blue', label='PPC')
    plt.fill_between(xs, mean_ppc-sem_ppc, mean_ppc+sem_ppc, color='tab:blue',alpha=0.3)
    #axs.axvline(stim2_on-start, color='darkgrey', linestyle='--')
    ax.axvspan(stim2_on-start, stim2_off-start, color='lightgrey')
    ax.axhline(y=0.5, color='black', linewidth=1)
    # plt.xlim([-1.1, 20+3])
    plt.ylim([0.4, 0.9])
    plt.xticks([-1,9,19,29])
    plt.plot(mean_pfc,  c='tab:purple', label='PFC')
    plt.fill_between(xs, mean_pfc-sem_pfc, mean_pfc+sem_pfc, color='tab:purple',alpha=0.3)
    plt.xlabel('from Cue_on')
    plt.ylabel('classifier accuracy')
    ax.spines[['right', 'top']].set_visible(False)

    plt.legend(fontsize=8)
    plt.savefig(fig_path1 + file_name + '_' + str(idx) + '_' + str(hp['sigma_x'])+'_'+str(hp['control_scale']) +'_'+generalize+ '.pdf')
    plt.show()
#
# #
for model_idx in np.array([50]):  # 6, 65,71
    for idx in np.array([15]):  # 23
        plot_decode_generalize(model_idx,idx,generalize='retro2prosp',control_scale=1)
        plot_decode_generalize(model_idx, idx, generalize='prosp2retro',control_scale=1)

        plot_decode_generalize(model_idx, idx, generalize='retro2prosp', control_scale=1.2)
        plot_decode_generalize(model_idx, idx, generalize='prosp2retro', control_scale=1.2)

        plot_decode_generalize(model_idx, idx, generalize='retro2prosp', control_scale=0.5)
        plot_decode_generalize(model_idx, idx, generalize='prosp2retro', control_scale=0.5)











