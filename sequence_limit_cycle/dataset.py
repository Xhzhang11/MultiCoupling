import torch
from torch.utils.data import Dataset, DataLoader

import task
import tools
import pdb

class TaskDataset(Dataset):

    def __init__(self, context_name, hp, mode='train', **kwargs):
        '''provide name of the rules'''
        self.context_name = context_name
        self.hp = hp
        # print('self.context_name',self.context_name)

        if mode == 'train':
            self.bach_size = hp['batch_size_train']
            self.task_mode = 'random'
        elif mode == 'test':
            self.bach_size = hp['batch_size_test']
            self.task_mode = 'random_validate'
        else:
            raise ValueError('Unknown mode: ' + str(mode))

        self.counter = 0

    def __len__(self):
        '''arbitrary'''
        return 10000000

    def __getitem__(self, index):

        self.trial = task.generate_trials(self.context_name, self.hp, self.task_mode, batch_size=self.bach_size,noise_on=True)

        '''model.x: trial.x,
                 model.y: trial.y,
                 model.cost_mask: trial.cost_mask,
                 model.seq_len: trial.seq_len,
                 model.initial_state: np.zeros((trial.x.shape[1], hp['n_rnn']))'''

        result = dict()
        result['inputs'] = torch.as_tensor(self.trial.x)
        result['target_outputs'] = torch.as_tensor(self.trial.y)
        result['cost_mask'] = torch.as_tensor(self.trial.cost_mask)
        result['cost_start_time'] = 0 # trial.cost_start_time
        result['cost_end_time'] = self.trial.max_seq_len
        result['seq_mask'] = tools.sequence_mask(self.trial.seq_len)
        result['initial_state'] = torch.zeros((self.trial.x.shape[1], self.hp['n_rnn']))
        result['initial_state_md'] = torch.zeros((self.trial.x.shape[1], self.hp['n_md']))
        result['epochs'] = self.trial.epochs

        if self.context_name=='dm_RT' or self.context_name=='dm_fixed':
            result['strength1'] = self.trial.strength1
            result['strength2'] = self.trial.strength2
        if self.context_name == 'dm_ctx' or self.context_name == 'multisensory':
            result['vis_strength1'] = self.trial.vis_strength1
            result['vis_strength2'] = self.trial.vis_strength2
            result['aud_strength1'] = self.trial.aud_strength1
            result['aud_strength2'] = self.trial.aud_strength2

        result['target_choice'] = self.trial.target_choice
        result['stim_duration'] = self.trial.stim_duration



        return result


class TaskDatasetForRun(object):

    def __init__(self, context_name, hp, noise_on=True, mode='test', **kwargs):
        '''provide name of the rules'''
        self.context_name = context_name
        self.hp = hp
        self.kwargs = kwargs
        self.noise_on = noise_on
        self.mode = mode
        #print('**self.mode',self.mode)

    def __getitem__(self):

        self.trial = task.generate_trials(self.context_name, self.hp, self.mode, noise_on=self.noise_on, **self.kwargs)
        #print('self.context_name',self.context_name)

        result = dict()
        result['inputs'] = torch.as_tensor(self.trial.x)
        result['target_outputs'] = torch.as_tensor(self.trial.y)
        result['cost_mask'] = torch.as_tensor(self.trial.cost_mask)
        result['cost_start_time'] = 0 # trial.cost_start_time
        result['cost_end_time'] = self.trial.max_seq_len
        result['seq_mask'] = tools.sequence_mask(self.trial.seq_len)
        result['initial_state'] = torch.zeros((self.trial.x.shape[1], self.hp['n_rnn']))
        result['initial_state_md'] = torch.zeros((self.trial.x.shape[1], self.hp['n_md']))
        result['epochs'] = self.trial.epochs
        result['target_choice'] = self.trial.target_choice


        # print('**',result['initial_state'])
        # pdb.set_trace()


        return result


