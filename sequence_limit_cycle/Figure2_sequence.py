import sys,os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import default
import tools
import lib_sequence


############ model


rule_name = 'DM'  # sys.argv[1]#'HL_task'
hp = default.get_default_hp(rule_name=rule_name)


hp['n_rnn'] = 128
hp['n_md'] = 128
hp['stim_duration'] = 600
hp['net'] = 'pfcmd'

hp['learning_rate']=0.0001
hp['activation']='softplus'
hp['activation_md']='sig'
hp['perturb_delay']=False
hp['root_path'] = os.path.abspath(os.path.join(os.getcwd(),"./"))






fig_path = hp['root_path']+'/Figures/'
# figure_path = os.path.join(fig_path, 'plot_activity_paper/')
figure_path = os.path.join(fig_path, 'plot_activity_paper/')
tools.mkdir_p(figure_path)


data_root = hp['root_path']+'/Datas/'
# figure_path = os.path.join(fig_path, 'plot_activity_paper/')
data_path = os.path.join(data_root, 'plot_activity_paper/')
tools.mkdir_p(data_path)

def get_model_dir_diff_context(model_idx, **kwargs):
    hp['type'] = kwargs['type']
    hp['tau'] = kwargs['tau']
    hp['model_idx'] = model_idx
    p_coh=0.9
    idx=1


    model_name = str(hp['type']) + '_'+str(hp['activation'])+ '_' \
                 + str(hp['n_rnn']) + '_' + str(hp['n_md'])+ '_lr'+str(hp['learning_rate'])\
               + '_bs' + str(hp['batch_size_train'])+ '_scale' + str(hp['scale_value'])\
                 + '_sw'+str(hp['sparsity_weight'])+ '_'+str(hp['freeze'])+ '_' + str(hp['model_idx'])

    # hp['model_dir_current'] = hp['root_path']+os.path.join('/'+'model_A2B_'+str(hp['p_coh']),  model_name)
    local_folder_name = os.path.join('/' + 'model_' + hp['rule_name'], model_name, str(idx))
    #local_folder_name = os.path.join('/' + 'model_' + str(hp['p_coh']), model_name, str(idx))

    model_dir = hp['root_path'] + local_folder_name+'/'
    #print(model_dir)
    #
    if os.path.exists(model_dir):
        print(f"The path '{model_dir}' exists.")
        #print("The path exists.")
    else:
        print(f"The path '{model_dir}' does not exist.")
        sys.exit(0)

    return model_dir, model_name




def plot_activity_pfc(model_idx,stim_delay,type):
    idx=1
    hp['rule_name'] = 'dm_fixed'
    hp['type'] = type
    hp['freeze'] = 'nofreeze'
    hp['scale_value'] = 1
    hp['sparsity_weight'] = 0.0

    hp['net'] = 'pfcmd'
    hp['rank'] = 1
    hp['n_rnn'] = 128
    hp['n_md'] = 128
    hp['stim_duration'] = 600
    hp['stim_delay'] = 300
    hp['learning_rate'] = 0.0001
    hp['batch_size'] = 64
    hp['batch_size_train'] = 64
    hp['scale_init'] = 0
    hp['tau'] = 80

    hp['initial_hh'] = 'zero'  # 'zero'#'Xavier'#'zero'#'Xavier'
    hp['stim_delay'] =stim_delay

    model_dir, model_name = get_model_dir_diff_context(model_idx=model_idx, type=hp['type'], tau=hp['tau'])
    hp['model_dir'] = model_dir


    lib_sequence.plot_activity_peak_order(figure_path,data_path,model_dir,hp)


for i in np.array([21]):  # 8,16,21,42,61
    plot_activity_pfc(model_idx=i,stim_delay=1800,type='type1')
















