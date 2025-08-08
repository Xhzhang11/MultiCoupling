import sys,os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
import scipy.stats as stats

import seaborn as sns
from scipy import stats
import json
import pandas as pd

import torch
from sklearn.decomposition import PCA
from matplotlib.patches import Circle

np.random.seed(0)
############ model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


hp = {}
hp['root_path'] = os.path.abspath(os.path.join(os.getcwd(),"./"))

hp['noise_var'] = 0.3
hp['seed'] = 0
hp['type'] = 'type0'
hp['rnn'] = 128
hp['md'] = 128
hp['batch_size'] = 64
hp['seq_len'] = 20
hp['npoints'] = 500
hp['tau'] = 100
hp['lr'] = 0.0001
hp['cere']=128
hp['if_cere']='yes'
hp['initial_hh'] = 'zero'
hp['sparsity']=0.8


data_root = hp['root_path']+'/Datas/'
data_path = os.path.join(data_root, 'dm_motor_cere_tanh_7.22_0/')





def get_model_dir_diff_context(model_idx):

    hp['model_idx'] = model_idx
    noise_var=1

    model_name = hp['type'] + '_n' + str(hp['noise_var']) + '_dealy' + str(200) \
                 + '_sl' + str(hp['seq_len']) + '_w' + str(hp['warpin']) \
                 + '_tail' + str(hp['n_tail']) + '_' + hp['task'] + '_' + str(hp['model_idx'])

    # hp['model_dir_current'] = hp['root_path']+os.path.join('/'+'model_A2B_'+str(hp['p_coh']),  model_name)
    local_folder_name = os.path.join('/' + 'models_saved', model_name)

    model_dir = hp['root_path'] + local_folder_name+'/'
    # print(model_dir)
    #
    if not os.path.exists(model_dir):
        print(f"The path '{model_dir}' does not exist.")
        sys.exit(0)

    return model_dir, model_name



def get_activity(hp, model_dir):

    hp['model_dir'] = model_dir

    model = network.CTRNN_PFCMD_Mod(hp['n_input'], hp['pfc'], 2, hp).to(device)
    model.load(model_dir=model_dir)
    trial = task.generate_trials_line(hp, batch_size=hp['batch_size'])
    x = torch.tensor(trial.x, dtype=torch.float32).to(device)

    _, hidden_states = model(x)

    # hidden_states = hidden_states.detach().cpu().numpy()

    print('hidden_states', hidden_states.shape)  # torch.Size([35, 64, 128])

    return hidden_states



def plot_activity(type,model_idx):
    hp['type']=type
    hp['if_cere']='yes'

    hp['noise_var'] = 0.3
    hp['delay']=200
    hp['seq_len']=20
    hp['npoints']=8
    hp['task']='line'
    model_dir, model_name = get_model_dir_diff_context(model_idx=model_idx)
    hidden_states = get_activity(hp, model_dir)

    mean_h = hidden_states.mean(dim=1).detach().cpu().numpy()# shape [T, H]
    print(mean_h.shape)#(35, 128)
    plt.figure(figsize=(10, 6))
    plt.plot(mean_h)
    plt.xlabel('Time Step')
    plt.ylabel('Hidden Unit Index')
    plt.title('Mean RNN Hidden Activity Across Batch')
    plt.show()
# plot_activity(type='type5',model_idx=0)


def get_circle_points(npoints, mag=10):
    angles = np.linspace(0, 2 * np.pi, npoints, endpoint=False)
    coords = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    return mag * coords
def plot_model(type, model_idx,cere2md,sparse,stim_MD):
    hp['type'] = type
    hp['if_cere'] = 'yes'
    hp['noise_var'] = 1
    hp['delay'] = 200
    hp['npoints'] = 8
    hp['task'] = 'line'

    hp['seq_len'] = 20
    hp['warpin']=0.9
    hp['n_tail']=1
    hp['sigma_rec'] = 0.1


    hp['stim_MD']=stim_MD
    hp['sigma_rec_cere']=0.25


    hp['scale_input']=1
    hp['cere2md']=cere2md
    hp['sparsity'] = 0

    # model_dir, model_name = get_model_dir_diff_context(model_idx=model_idx)
    # model = network.CTRNN_PFCMD_Mod(hp['n_input'], hp['pfc'], 2, hp).to(device)
    # model.load(model_dir=model_dir)
    #
    # trial = task.generate_trials_line(hp, batch_size=hp['batch_size'])
    # x = torch.tensor(trial.x, dtype=torch.float32).to(device)
    # y = torch.tensor(trial.y, dtype=torch.float32).to(device)
    #
    # with torch.no_grad():
    #     pred, hidden_states = model(x)
    #
    #
    #
    #
    # y_pred = pred.detach().cpu().numpy()
    # y_true = y.detach().cpu().numpy()
    #
    #
    # np.save(data_path+'y_pred_stim'+str(stim_MD)+'.npy',y_pred)
    # np.save(data_path+'y_true_stim'+str(stim_MD)+'.npy',y_true)

    y_pred = np.load(data_path + 'y_pred_stim' + str(stim_MD) + '.npy')
    y_true = np.load(data_path + 'y_true_stim' + str(stim_MD) + '.npy')




    print('y_pred',y_pred.shape)
    print('y_true', y_true.shape)

    coords = get_circle_points(hp['npoints'])
    print('coords',coords,coords.shape)

    # Plot
    fig = plt.figure(figsize=(4, 4))
    final_radius = 0.8
    for i in range(y_pred.shape[1]):
        #plt.plot(y_true[:, i, 0], y_true[:, i, 1], 'k--')                   # Ground truth
        # plt.plot(y_true[-1, i, 0], y_true[-1, i, 1], 'o', color='k')        # Ground truth endpoint
        plt.plot(y_pred[:, i, 0], y_pred[:, i, 1], color='grey',alpha=0.7)
        plt.plot(y_pred[0, i, 0], y_pred[0, i, 1],'o', color='green')  # Predicted path

        plt.plot(y_pred[-1, i, 0], y_pred[-1, i, 1], 'o', color='tab:orange')  # Predicted endpoint
    for i in range(coords.shape[0]):
        end_x, end_y = coords[i, 0], coords[i, 1]
        circle = Circle((end_x, end_y), final_radius,edgecolor='blue', facecolor='none', lw=1.5)
        plt.gca().add_patch(circle)

    plt.axis('equal')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    # plt.grid()
    # title = 'cere2md_'+str(cere2md)+'_sparse'+str(sparse)+'_stim'+str(stim_MD)
    # plt.title('cere2md='+str(cere2md)+'; sparse'+str(sparse)+'; stim'+str(stim_MD))
    # fig.savefig(figure_path +model_name+'stm'+str(stim_MD)+ '.pdf')

    plt.show()



np.random.seed(0)
hp['rng'] = np.random

# plot_model(type='type4', model_idx=0,cere2md=1,sparse=0,stim_MD=1)
for stim in np.array([1.0,1.1,1.2,1.5]):#1.0,1.1,1.2,1.5
    plot_model(type='type4', model_idx=5, cere2md=1,sparse=0,stim_MD=stim)



