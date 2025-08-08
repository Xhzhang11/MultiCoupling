import numpy as np
import os
import task



def get_default_hp(rule_name, random_seed=None):
    '''Get a default hp.

    Useful for debugging.

    Returns:
        hp : a dictionary containing training hpuration
    '''
    root_path = os.path.abspath(os.path.join(os.getcwd(),"../"))
    save_dir = os.path.join(root_path, 'models_saved')
    figure_dir = os.path.join(root_path, 'z_figure')
    print('root_path',root_path)


    n_input = 1+2
    n_output = 2

    if rule_name == 'dm_ctx':
        n_input = 1 + 4 + 2
    if rule_name == 'multisensory':
        n_input = 1 + 4
        n_output = 2

    if rule_name == 'GoNogo':
        n_output = 1

    # default seed of random number generator
    if random_seed is None:
        seed = np.random.randint(10000000)
    else:
        seed = random_seed


    hp = {
        'root_path':root_path,
        'figure_dir': figure_dir,

        'e_prop':0.8,
        'rule_name': 'dm',
        # batch size for training
        'batch_size_train': 512, #128,#64,
        'lsq1':0.999,
        'lsq2': 0.000,

        # Optimizer adam or sgd
        'optimizer': 'adam',
        # Type of activation functions: relu, softplus
        'activation': 'softplus',
        # Time constant (ms)
        'tau': 100,
        # discretization time step (ms)
        'dt': 20,
        # discretization time step/time constant
        'alpha': 1,
        # initial standard deviation of non-diagonal recurrent weights
        'initial_std': 0.3,#0.25,#0.27,#0.3,
        # recurrent noise
        'sigma_rec': 0.01,
        # input noise
        'sigma_x': 0.1,
        # a default weak regularization prevents instability
        'l1_firing_rate': 0.1,
        # l2 regularization on activity
        'l2_firing_rate': 0.1,
        # l1 regularization on weight
        'l1_weight': 0.0,
        # l2 regularization on weight
        'l2_weight': 0.1,
        # number of input units
        'n_input': n_input,
        # number of output units
        'n_output': n_output,
        # number of recurrent units
        'n_rnn':256,
        'n_md':30,
        # learning rate
        'learning_rate': 0.0005,
        # random number generator
        'seed': seed,
        'rng': np.random.RandomState(seed),

        'p_coh':1,
        'response_time':100,

        'sparsity_HL': 0.0,

        'dropout':0.0,



    }

    return hp
