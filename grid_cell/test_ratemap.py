
import sys, os
import numpy as np

import ratemaps

sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"../..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"../")))





from matplotlib import pyplot as plt









#load parames
hp = {}
hp['root_path'] = os.path.abspath(os.path.join(os.getcwd(),"./"))


hp['Ng']=256
hp['Np'] = 256
hp['rng'] = np.random.RandomState(1)


hp['learning_rate']=0.0001
hp['batch_size_test']=5000

hp['rotate'] = True
hp['sparse_EC_CA1']=0
hp['sparse_CA3_CA1']=0
hp['is_EI'] = 'EIno'
hp['mode'] = 'test'
hp['box_width'] = 2.6
# hp['degree'] = 90#0,30,45,60,90

hp['save_dir'] = hp['root_path']+'/model'
print('save_dir', hp['save_dir'])

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

def plot_grid(wDG,wc,wa,model_idx,scale,env):
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


    fig_path = hp['root_path'] + '/Datas/'
    figure_path = os.path.join(fig_path, 'test_ratemap'+'/')




    fig_path_1 = os.path.join(figure_path,  str(hp['wDG'])+ '_a3w' + str(hp['wc'])+ '_a1w' + str(hp['wa'])+'_m'+ str(model_idx)+'/')



    # for idx in np.array([0]):
    #     plt.plot(place_cell .c_recep_field[idx, 0], place_cell .c_recep_field[idx, 1], 'o', markersize=3,
    #              label=str(idx))
    # plt.xticks([-hp['box_width'] / 2, hp['box_width'] / 2])
    # plt.yticks([-hp['box_width'] / 2, hp['box_width'] / 2])
    # plt.legend()
    # plt.show()

    # img




    for get_grid in np.array(['EC','CA3','DG','CA1']):  # ['EC','CA3','DG','CA1']
        if hp['env'] == 'circle':
            ratemaps.Plot_ratemap_manipulate_weight(get_grid, hp, scale,data_path=fig_path_1)



        if hp['env'] == 'rectangle':
            ratemaps.Plot_ratemap_manipulate_weight_rectangle(get_grid, hp,  scale,data_path=fig_path_1)



for idx in range(105,106):
    print('==========================================================',idx)
    plot_grid(wDG=0.4, wc=1.0, wa=0.3, model_idx=idx, scale=1.0, env='circle')  ####
    plot_grid(wDG=0.4, wc=1.0, wa=0.3, model_idx=idx, scale=1.0, env='rectangle')  ####


























