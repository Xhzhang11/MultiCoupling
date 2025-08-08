import sys,os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ranksums
from scipy.stats import wilcoxon

import default
import tools
import state_space_lib
from scipy.io import savemat




hp=default.get_default_hp(rule_name='prosp')

hp['rng']=np.random.RandomState(0)
######

#=========================  plot =============================

hp['root_path'] = os.path.abspath(os.path.join(os.getcwd(), "./"))

fig_path = hp['root_path']+'/Figures/'
figure_path = os.path.join(fig_path, 'state_pca/')
tools.mkdir_p(figure_path)
data_path = os.path.join(hp['root_path'], 'Datas','state_pca/')
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



def calculate_angle_example1(model_idx,idx,seed):
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


    model_dir, model_name = get_model(model_idx, idx)
    epoch = state_space_lib.get_epoch(model_dir, rule_name='retro', hp=hp)
    fig_path_0 = os.path.join(figure_path, model_name)
    tools.mkdir_p(fig_path_0)

    print('model_name',model_name)

    hp['plot_activtity'] = True#False#
    hp['in_strength']=1.0
    hp['seed'] = seed


    state_space_lib.PCA_plot_3D_selected_angle_pfc_example1(figure_path,data_path, model_name, model_dir, idx, hp,rule_name='retro',
                                                                   start_proj=epoch['stim1_off'][0],
                                                                   end_proj=epoch['stim2_off'][0])




calculate_angle_example1(model_idx=50,idx=15,seed=4)







