import numpy as np
import os
import task


def input_output_n(rule_name):
    # basic timing tasks
    cue_dim = 1

    if rule_name == 'retro':
        return 12+cue_dim, 12
    elif rule_name == 'spatial_comparison2':
        return 12+cue_dim, 12

    elif rule_name == 'spatial_reproduction':
        return 12+1, 12

def get_default_hp(rule_name, random_seed=None):
    '''Get a default hp.

    Useful for debugging.

    Returns:
        hp : a dictionary containing training hpuration
    '''

    root_path = os.path.abspath(os.path.join(os.getcwd(), "../"))
    save_dir = os.path.join(root_path, 'models_saved')
    figure_dir = os.path.join(root_path, 'z_figure')
    root_path_1 = os.path.abspath(os.path.join(os.getcwd(), "."))
    picture_dir = os.path.join(root_path_1, 'picture3/')

    root_path_data = os.path.abspath(os.path.join(os.getcwd(), "../"))



    # default seed of random number generator
    if random_seed is None:
        seed = np.random.randint(100)
    else:
        seed = random_seed

    hp = {
        'n_eachring': 64,
        'n_cue':1,
        'n_rule':2,
        'rule_name': rule_name,

        'root_path':root_path,
        'picture_dir':picture_dir,
        'figure_dir': figure_dir,

        'e_prop':0.8,

        # batch size for training
        'batch_size_train': 512, #128,#64,
        # batch_size for testing
        'batch_size_test': 512,
        # Type of RNNs: RNN
        'rnn_type': 'RNN',
        # Optimizer adam or sgd
        'optimizer': 'adam',
        # Type of activation functions: relu, softplus
        'activation': 'softplus',
        # Time constant (ms)
        'tau':20,
        'dt':10,


        # discretization time step/time constant
        'alpha': 1,
        # initial standard deviation of non-diagonal recurrent weights
        'initial_std': 0.3,#0.25,#0.27,#0.3,
        # recurrent noise
        'sigma_rec': 0.05,
        # input noise
        'sigma_x': 0.01,
        # a default weak regularization prevents instability
        'l1_firing_rate': 0,
        # l2 regularization on activity
        'l2_firing_rate': 0.1,
        # l1 regularization on weight
        'l1_weight': 0.1,
        # l2 regularization on weight
        'l2_weight': 0.0,
        # Type of loss functions
        'loss_type': 'lsq',

        'n_input': 0,
        'n_output': 0,

        # number of recurrent units
        'n_rnn':512,
        'n_md':64,
        # learning rate
        'learning_rate': 0.0001,
        # random number generator
        'seed': seed,
        'rng': np.random.RandomState(seed),
        'is_EI': 'EIno',
        'use_reset':'yes',
        'stim_delay':100,
        'cue_delay':100,
        'stim':100,
        'cuestim':100,
        'resp':40,

        'response':40,
        'model_idx':0,
        'dropout':0.0,
        'mask_type':'no',
        'mode_mask':'train',
        'sd_gaussianline':8,
        'in_strength':1,
        'control_scale':1






    }

    return hp
