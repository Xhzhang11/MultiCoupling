"""Collections of tasks."""

from __future__ import division
import numpy as np
import math
import sys
import tools

import pdb
def get_dist(original_dist):
    '''Get the distance in periodic boundary conditions'''
    return np.minimum(abs(original_dist), 2*np.pi-abs(original_dist))


# generating values of x, y, c_mask specific for tasks
# config contains hyper-parameters used for generating tasks
class Trial(object):
    """Class representing a batch of trials."""

    def __init__(self, config, xtdim, batch_size):
        """A batch of trials.

        Args:
            config: dictionary of configurations
            xtdim: int, number of total time steps
            batch_size: int, batch size
        """
        self.float_type = 'float32'  # This should be the default
        self.config = config
        self.dt = self.config['dt']

        self.n_input = self.config['n_input']
        self.n_output = self.config['n_output']

        self.batch_size = batch_size
        self.xtdim = xtdim

        # time major
        self.x = np.zeros((xtdim, batch_size, self.n_input), dtype=self.float_type)
        self.y = np.zeros((xtdim, batch_size, self.n_output), dtype=self.float_type)

        self.cost_mask = np.zeros((xtdim, batch_size, self.n_output), dtype=self.float_type)
        # strength of input noise
        self._sigma_x = config['sigma_x'] * math.sqrt(2./self.config['alpha'])

    def expand(self, var):
        """Expand an int/float to list."""
        if not hasattr(var, '__iter__'):
            var = [var] * self.batch_size
        return var

    def add(self, loc_type, loc_idx, ons, offs, strengths):
        """Add an input or stimulus output to the indicated channel.

        Args:
            loc_type: str type of information to be added
            loc_idx: index of channel
            ons: int or list, index of onset time
            offs: int or list, index of offset time
            strengths: float, strength of input or target output
            gaussian_center: float. location of gaussian bump, only used if loc_type=='line_gaussian_input' or 'line_gaussian_output'
        """

        for i in range(self.batch_size):
            if loc_type == 'input':
                self.x[ons[i]: offs[i], i, loc_idx] = strengths[i]


            elif loc_type == 'out':
                self.y[ons[i]: offs[i], i, loc_idx] = strengths[i]
            elif loc_type == 'cost_mask':
                self.cost_mask[ons[i]: offs[i], i, loc_idx] = strengths[i]
            else:
                raise ValueError('Unknown loc_type')

    def add_x_noise(self):
        # print('self.x',self.x.shape)
        self.x[:,:,1:] += self.config['rng'].randn(*self.x[:,:,1:].shape) * self._sigma_x



def _dm_RT(config, mode, **kwargs):
    '''
    Two stimuli are presented, a pulse cue is then input to the network at the very end of the presentation.
    The network should respond the index of the stronger or weaker stimulus, depending on the content of the pulse cue.
    '''
    dt = config['dt']
    rng = config['rng']
    batch_size = kwargs['batch_size']

    fix_on = (rng.uniform(0, 0, batch_size) / dt).astype(int)
    stim1_on = (rng.uniform(0, 0, batch_size) / dt).astype(int)

    # stim_duration = int(config['stim_duration'] / dt)
    stim_duration = int(rng.uniform(400, 400) / dt)
    stim1_off = stim1_on + stim_duration

    stim_delay = 0#int(config['stim_delay'] / dt)
    response_on = stim1_off + stim_delay
    response_duration = int(config['response_time'] / dt)

    response_off = response_on + response_duration
    xtdim = response_off


    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']
        gamma_bar = rng.uniform(0.5, 0.5, batch_size)
        c = 0.005*rng.choice([-6.4, -12.8, -25.6, -51.2, 6.4, 12.8, 25.6, 51.2], (batch_size,))
        #c = 0.005 * rng.choice([-25.6,25.6], (batch_size,))
        # print('c',c)

        strength1 = gamma_bar + c
        strength2 = gamma_bar - c



    elif mode == 'random_validate':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        gamma_bar = rng.uniform(0.5, 0.5, batch_size)
        c = 0.005*rng.choice([-12.8, -25.6, -51.2, 12.8, 25.6, 51.2], (batch_size,))
        strength1 = gamma_bar + c
        strength2 = gamma_bar - c
        #c = 0.005 * rng.choice([-25.6,25.6], (batch_size,))

    elif mode == 'test':
        batch_size = kwargs['batch_size']

        gamma_bar = kwargs['gamma_bar']
        if not hasattr(gamma_bar, '__iter__'):
            gamma_bar = np.array([gamma_bar] * batch_size)

        c = kwargs['c']
        if not hasattr(c, '__iter__'):
            c = np.array([c] * batch_size)

        choice = kwargs['choice']
        if not hasattr(choice, '__iter__'):
            choice = np.array([choice] * batch_size)

        strength1 = gamma_bar + c
        strength2 = gamma_bar - c


    elif mode == 'pca':
        batch_size = kwargs['batch_size']
        gamma_bar = rng.uniform(0.5, 0.5, batch_size)

        c = kwargs['c']
        if not hasattr(c, '__iter__'):
            c = np.array([c] * batch_size)

        choice = kwargs['choice']
        if not hasattr(choice, '__iter__'):
            choice = np.array([choice] * batch_size)

        strength1 = gamma_bar + c
        strength2 = gamma_bar - c


    else:
        raise ValueError('Unknown mode: ' + str(mode))





    trial = Trial(config, xtdim.max(), batch_size)


    trial.add('input', 0, ons=fix_on, offs=response_on, strengths=trial.expand(0.))
    # input 0
    trial.add('input', 1, ons=stim1_on, offs=stim1_off, strengths=strength1)
    # input 1
    trial.add('input', 2, ons=stim1_on, offs=stim1_off, strengths=strength2)
    # output


    output_target1 = 1. * (strength1 > strength2)
    output_target2 = 1. * (strength1 <= strength2)

    trial.add('out', 0, ons=response_on, offs=response_off, strengths=output_target1)
    trial.add('out', 1, ons=response_on, offs=response_off, strengths=output_target2)

    target_choice = 0. * (strength1 >strength2)+1. * (strength1 <=strength2)#np.sign(strength1 - strength2)

    trial.add('cost_mask', 0, ons=stim1_on, offs=response_off, strengths=trial.expand(5.))
    trial.add('cost_mask', 1, ons=stim1_on, offs=response_off, strengths=trial.expand(5.))
    trial.add('cost_mask', 0, ons=response_on, offs=response_off, strengths=trial.expand(5.))
    trial.add('cost_mask', 1, ons=response_on, offs=response_off, strengths=trial.expand(5.))


    trial.epochs = {'fix': (fix_on, stim1_on),
                    'stim': (stim1_on, stim1_off),
                    'response': (response_on, response_off)}


    trial.stim_duration = stim_duration
    trial.strength1 = strength1
    trial.strength2 = strength2
    trial.target_choice = target_choice

    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()

    return trial


def _dm_fixed(config, mode, **kwargs):
    '''
    Two stimuli are presented, a pulse cue is then input to the network at the very end of the presentation.
    The network should respond the index of the stronger or weaker stimulus, depending on the content of the pulse cue.
    '''
    dt = config['dt']
    rng = config['rng']
    batch_size = kwargs['batch_size']

    fix_on = (rng.uniform(0, 0, batch_size) / dt).astype(int)
    stim1_on = (rng.uniform(0, 0, batch_size) / dt).astype(int)
    stim_duration = int(config['stim_duration'] / dt)
    stim1_off = stim1_on + stim_duration

    stim_delay = int(config['stim_delay'] / dt)
    response_on = stim1_off + stim_delay
    response_duration = int(config['response_time'] / dt)

    response_off = response_on + response_duration
    xtdim = response_off

    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']
        gamma_bar = rng.uniform(0.5, 0.5, batch_size)
        c = 0.005 * rng.choice([-12.8, -25.6, -51.2, 12.8, 25.6, 51.2], (batch_size,))


    elif mode == 'random_validate':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        gamma_bar = rng.uniform(0.5, 0.5, batch_size)
        c = 0.005 * rng.choice([-12.8, -25.6, -51.2, 12.8, 25.6, 51.2], (batch_size,))


    elif mode == 'test1':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        gamma_bar = rng.uniform(0.5, 0.5, batch_size)
        #c = 0.005 * rng.choice([-6.4, -12.8, -25.6, -51.2, 6.4, 12.8, 25.6, 51.2], (batch_size,))
        #c = 0.005 * rng.choice([6.4, 12.8, 25.6, 51.2], (batch_size,))
        c = 0.005 * rng.choice([-25.6, -51.2, 25.6, 51.2], (batch_size,))


    elif mode == 'pca':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        gamma_bar = rng.uniform(0.5, 0.5, batch_size)
        #c = 0.005 * rng.choice([-6.4, -12.8, -25.6, -51.2, 6.4, 12.8, 25.6, 51.2], (batch_size,))
        #c = 0.005 * rng.choice([6.4, 12.8, 25.6, 51.2], (batch_size,))
        c = 0.005 * rng.choice([-51.2,  51.2], (batch_size,))


    elif mode == 'perf':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        gamma_bar = rng.uniform(0.5, 0.5, batch_size)
        #c = 0.005 * rng.choice([-6.4, -12.8, -25.6, -51.2, 6.4, 12.8, 25.6, 51.2], (batch_size,))
        #c = 0.005 * rng.choice([6.4, 12.8, 25.6, 51.2], (batch_size,))
        c = 0.005 * rng.choice([-12.8, -25.6, -51.2, 12.8, 25.6, 51.2], (batch_size,))





    elif mode == 'test':
        batch_size = kwargs['batch_size']

        gamma_bar = kwargs['gamma_bar']
        if not hasattr(gamma_bar, '__iter__'):
            gamma_bar = np.array([gamma_bar] * batch_size)

        c = kwargs['c']
        if not hasattr(c, '__iter__'):
            c = np.array([c] * batch_size)

        choice = kwargs['choice']
        if not hasattr(choice, '__iter__'):
            choice = np.array([choice] * batch_size)



    elif mode == 'pca':
        batch_size = kwargs['batch_size']
        gamma_bar = rng.uniform(0.5, 0.5, batch_size)
        c = kwargs['c']
        # c = 0.005 * rng.choice([-25.6, -51.2,  25.6, 51.2], (batch_size,))




    else:
        raise ValueError('Unknown mode: ' + str(mode))

    strength1 = gamma_bar + c
    strength2 = gamma_bar - c

    trial = Trial(config, xtdim.max(), batch_size)

    trial.add('input', 0, ons=fix_on, offs=response_on, strengths=trial.expand(1.))
    # trial.add('input', 0, ons=fix_on, offs=stim1_off, strengths=trial.expand(1.))

    # input 0
    trial.add('input', 1, ons=stim1_on, offs=stim1_off, strengths=strength1)
    # input 1
    trial.add('input', 2, ons=stim1_on, offs=stim1_off, strengths=strength2)
    # output
    if config['perturb_delay']:
        trial.add('input', 1, ons=stim1_off+1, offs=stim1_off+4, strengths=trial.expand(2))
        trial.add('input', 2, ons=stim1_off+1, offs=stim1_off+4, strengths=trial.expand(2))






    output_target1 = 1. * (strength1 > strength2)
    output_target2 = 1. * (strength1 <= strength2)

    trial.add('out', 0, ons=response_on, offs=response_off, strengths=output_target1)
    trial.add('out', 1, ons=response_on, offs=response_off, strengths=output_target2)

    target_choice = 0. * (strength1 > strength2) + 1. * (strength1 <= strength2)  # np.sign(strength1 - strength2)

    trial.add('cost_mask', 0, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))
    trial.add('cost_mask', 1, ons=stim1_on, offs=response_off, strengths=trial.expand(1.))
    trial.add('cost_mask', 0, ons=response_on, offs=response_off, strengths=trial.expand(1.))
    trial.add('cost_mask', 1, ons=response_on, offs=response_off, strengths=trial.expand(1.))

    trial.epochs = {'fix': (fix_on, stim1_on),
                    'stim': (stim1_on, stim1_off),
                    'response': (response_on, response_off)}

    trial.stim_duration = stim_duration
    trial.strength1 = strength1
    trial.strength2 = strength2
    trial.target_choice = target_choice

    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()

    return trial






def dm_RT(config, mode, **kwargs):
    return _dm_RT(config, mode, **kwargs)

def dm_fixed(config, mode, **kwargs):
    return _dm_fixed(config, mode, **kwargs)

# map string to function
rule_mapping = {
'dm_RT': dm_RT,
'dm_fixed': dm_fixed,


               }

def generate_trials(context_name, hp, mode, noise_on=True, **kwargs):
    """Generate one batch of data.

    Args:
        hp: dictionary of hyperparameters
        mode: str, the mode of generating. Options: random, test, psychometric
        noise_on: bool, whether input noise is given

    Return:
        trial: Trial class instance, containing input and target output
    """
    # print(rule)
    config = hp
    kwargs['noise_on'] = noise_on

    seed =config['seed']
    rng = config['rng']
    trial = rule_mapping[context_name](config, mode, **kwargs)

    batch_size =  kwargs['batch_size']
    #print('****** scale_RDM',scale_RDM)

    if noise_on:
        trial.add_x_noise()
    # print('@@@@@HL_task trial.x',trial.x[:,0,:].shape,'\n', np.around(trial.x[:,0,:],decimals=2))

    return trial


# test the generation of trials
if __name__ == "__main__":
    import seaborn as sns
    from matplotlib import pyplot as plt

    import default

    print(sys.argv)
    print(len(sys.argv))

    rule_name = 'dm_fixed'
    hp = default.get_default_hp(rule_name)


    hp['stim_duration'] = 200
    hp['stim_delay'] = 100
    hp['response_time'] = 40
    hp['fix_value'] = 0.0
    hp['gamma_value']=0.7
    hp['same_input']=False
    hp['sigma_x']=0.1
    hp['perturb_delay']=False

    # hp['cue_delay'] = 100
    # hp['stim'] = 40
    # hp['response_time'] = 40
    #
    # hp['sparsity_HL'] = 0.5
    # hp['p_coh'] = 0.8

    trial = generate_trials(rule_name, hp, 'random_validate', batch_size=5, noise_on=True)

    for i_batch in range(5):#,'HL_task'
        print(rule_name, '********* batch:',i_batch)
        print('==target_choice: ', trial.target_choice[i_batch])

        print('@@@@@trial.x',trial.x[:,i_batch,:].shape,'\n', np.around(trial.x[:,i_batch,:],decimals=2))
        print('@@@@@trial.y',trial.y[:,i_batch,:].shape,'\n', np.around(trial.y[:,i_batch,:],decimals=2))

