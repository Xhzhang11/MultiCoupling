"""Utility functions."""

import os
import errno
import json
import pickle
import numpy as np
import torch

import default


def load_hp(model_dir):
    """Load the hyper-parameter file of model save_name"""
    fname = os.path.join(model_dir, 'hp.json')

    if os.path.isfile(fname):
        with open(fname, 'r') as f:
            hp = json.load(f)
    else:
        hp = default.get_default_hp(rule_name='both_RDM_HL_task')
        #print('hp.json:',fname)

    hp['seed'] = np.random.randint(0, 1000000)
    hp['rng'] = np.random.RandomState(hp['seed'])
    return hp


def save_hp(hp, model_dir):
    """Save the hyper-parameter file of model save_name"""
    hp_copy = hp.copy()
    hp_copy.pop('rng')  # rng can not be serialized
    with open(os.path.join(model_dir, 'hp.json'), 'w') as f:
        json.dump(hp_copy, f)


def mkdir_p(path):
    """
    Portable mkdir -p

    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_log(log, log_name='log.json'):
    """Save the log file of model."""
    model_dir = log['model_dir']
    fname = os.path.join(model_dir, log_name)
    with open(fname, 'w') as f:
        json.dump(log, f)


def splid_model_name(model_name,start_str):
        # Split by underscores
    parts = model_name.split('_')

    # Find index where 'pc' prefix starts
    start_idx = next(i for i, p in enumerate(parts) if p.startswith(start_str))

    # Join the rest
    suffix = '_'.join(parts[start_idx:])

    return suffix


def load_log(model_dir, log_name='log.json'):
    """Load the log file of model save_name"""
    fname = os.path.join(model_dir, log_name)
    if not os.path.isfile(fname):
        return None

    with open(fname, 'r') as f:
        log = json.load(f)
    return log


def load_pickle(file):
    try:
        with open(file, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print('Unable to load data ', file, ':', e)
        raise
    return data


def sequence_mask(lens):
    '''
    Input: lens: numpy array of integer

    Return sequence mask
    Example: if lens = [3, 5, 4]
    Then the return value will be
    tensor([[1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0]], dtype=torch.uint8)
    :param lens:
    :return:
    '''
    max_len = max(lens)
    # return torch.arange(max_len).expand(len(lens), max_len) < lens.unsqueeze(1)
    return torch.t(torch.arange(max_len).expand(len(lens), max_len) < torch.tensor(np.expand_dims(lens, 1), dtype=torch.float32))


def elapsed_time(totaltime):
    hrs  = int(totaltime//3600)
    mins = int(totaltime%3600)//60
    secs = int(totaltime%60)

    return '{}h {}m {}s elapsed'.format(hrs, mins, secs)



def generate_sparsity_matrix(rows=0,cols=0,sparsity=0):
    import numpy as np
    from scipy.sparse import coo_matrix

    # Define the size of the matrix
    rows, cols = rows, cols

    # Set the desired sparsity level (e.g., 0.2 for 20% sparsity)
    sparsity_level = sparsity

    # Calculate the number of zero elements
    num_zero_elements = int(rows * cols * sparsity_level)

    # Generate random indices for zero elements
    zero_indices = np.random.choice(rows * cols, size=num_zero_elements, replace=False)

    # Create a sparse matrix with zeros at the selected indices
    values = np.ones(num_zero_elements)  # Use ones for non-zero values
    sparse_matrix = coo_matrix((values, (zero_indices // cols, zero_indices % cols)), shape=(rows, cols))

    # print("Sparse Matrix with Sparsity Level {:.2%}:".format(sparsity_level))
    # print(sparse_matrix)
    dense_matrix = sparse_matrix.toarray()
    # print(dense_matrix)

    return dense_matrix




if __name__ == '__main__':


    hp = {}
    hp['e_prop'] = 0.8
    hp['sparsity']=0.8
    n_rnn = 10

    mask = mask_hh_1cross(hp, n_rnn)
    print('mask','\n',mask)