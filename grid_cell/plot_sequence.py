
import sys, os
import numpy as np
import torch


sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"../..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"../")))


import sequence_lib



#load parames
hp = {}
hp['root_path'] = os.path.abspath(os.path.join(os.getcwd(),"./"))

hp['Ng']=512

# hp['rng'] = np.random.RandomState(1)

hp['batch_size_test']=512
hp['batch_size']=512
hp['is_EI'] = 'EIno'
hp['wDG'] = 0.4
hp['wc'] = 1.0
hp['wa'] = 0.5
hp['mode'] = 'test'
hp['env'] = 'circle'
hp['activation'] = 'relu'
hp['activation_md'] = 'relu'
hp['activation_CA3'] = 'sigmoid'
hp['type'] = 'type1'
hp['freeze'] = 'no'
hp['sparsity'] = 0.5
hp['init'] = 'rand'
hp['dt'] = 20
hp['Np'] = 512
hp['k'] = 1
hp['learning_rate'] = 0.00005
hp['surround_scale'] = 2
hp['place_cell_rf']=0.12

hp['periodic']=False
hp['DoG']=True

hp['box_width']=2.6                # width of training environment
hp['box_height']=2.6



rng = np.random#.RandomState(0)

hp['rng'] = rng



fig_path = hp['root_path'] + '/Figures/'
figure_path = os.path.join(fig_path, 'plot_sequence' + '/')


data_root = hp['root_path'] + '/Datas/'
data_path = os.path.join(data_root, 'gridcell_dffinput_5.12_k1_circle_even_alpha' + '/')

def get_model_dir_diff_context(model_idx):
    sl=10
    hp['run_ID'] = hp['type']+'_tau'+str(30)+'_speed'+str(hp['speed'])+ '_Np' + str(hp['Np'])+ '_sl' + \
				   str(10)+'_lr'+str(hp['learning_rate'])+'_wa'+str(hp['wa'])\
				   +'_k'+str(hp['k'])+'_0_'+ str(model_idx)

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




def plot_sequence_speed0_diff_init(model_idx,sl,tau):

    data_path_0 = os.path.join(data_path, 'plot_sequence' + '/')





    hp['tau']=tau
    hp['model_idx'] = model_idx
    hp['sequence_length'] = sl

    hp['speed'] = 5
    hp['same_start'] = False

    hp['sigma_rec'] = 0.04

    # hp['new_env'] = True
    model_dir, run_ID = get_model_dir_diff_context(model_idx=model_idx)

    hp['run_ID']=run_ID

    fig_path_1 = os.path.join(figure_path, run_ID + '/')


    hp['get_grid']='CA1'



    sequence_lib.plot_sequence_and_cell_activity_speed0_tau25(data_path_0,hp,fig_path=fig_path_1,
                                               speed_scale=0.01)




hp['diff_seed']=True
for seed in range(100):
    hp['seed']=seed
    plot_sequence_speed0_diff_init(model_idx=100, sl=14, tau=20)  ####
    plot_sequence_speed0_diff_init(model_idx=100, sl=17, tau=25)  ####



