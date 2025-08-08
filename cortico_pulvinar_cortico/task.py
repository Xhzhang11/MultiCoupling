"""Collections of tasks."""

from __future__ import division
import numpy as np
import math
import sys


def get_dist(original_dist):
    '''Get the distance in periodic boundary conditions'''
    return np.minimum(abs(original_dist),2*np.pi-abs(original_dist))


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

        self.n_eachring = self.config['n_eachring']
        self.n_guassianline = self.config['n_eachring']
        self.sd_gaussianline = self.config['sd_gaussianline']


        self.pref_line_gaussian = np.arange(0,2*np.pi,2*np.pi/self.n_eachring) # preferences



    def line_gaussian_activity(self, x_loc):
        """Input activity given location."""
        #print('#### self.pref', self.pref_line_gaussian * 180 / np.pi)
        # print('@@@@@ x_loc', x_loc*180/np.pi)

        dist = get_dist(x_loc - self.pref_line_gaussian)  # periodic boundary
        dist /= np.pi / self.sd_gaussianline
        return 1.0 * np.exp(-dist ** 2 / 2)

    def add_y_loc(self, y_loc):
        """Target response given location."""
        dist = get_dist(y_loc-self.pref_line_gaussian)  # periodic boundary
        dist /= np.pi / self.sd_gaussianline
        y = 1.0 * np.exp(-dist ** 2 / 2)

        return y


    def expand(self, var):
        """Expand an int/float to list."""
        if not hasattr(var, '__iter__'):
            var = [var] * self.batch_size
        return var

    def add(self, loc_type, loc_idx, ons, offs, strengths, gaussian_center=None):
        """Add an input or stimulus output to the indicated channel.

        Args:
            loc_type: str type of information to be added
            loc_idx: index of channel
            ons: int or list, index of onset time
            offs: int or list, index of offset time
            strengths: float, strength of input or target output
            gaussian_center: float. location of gaussian bump, only used if loc_type=='line_gaussian_input' or 'line_gaussian_output'
        """

        if loc_type == 'cue_input':
            for i in range(self.batch_size):
                self.x[ons[i]: offs[i], i, loc_idx] = strengths[i]

        elif loc_type == 'out':

            for i in range(self.batch_size):
                self.y[ons[i]: offs[i], i, loc_idx] = strengths[i]

        elif loc_type == 'cost_mask':

            for i in range(self.batch_size):
                self.cost_mask[ons[i]: offs[i], i, loc_idx] = strengths[i]

        elif loc_type == 'line_gaussian_input':

            loc_start_idx = loc_idx
            loc_end_idx = loc_idx + self.config['n_eachring']

            for i in range(self.batch_size):
                self.x[ons[i]: offs[i], i, loc_start_idx:loc_end_idx] += self.line_gaussian_activity(gaussian_center[i]) * strengths[i]

        elif loc_type == 'line_gaussian_output':

            loc_start_idx = loc_idx
            loc_end_idx = loc_idx + self.config['n_eachring']

            for i in range(self.batch_size):
                self.y[ons[i]: offs[i], i, loc_start_idx:loc_end_idx] += self.add_y_loc(gaussian_center[i]) * strengths[i]

        else:
            raise ValueError('Unknown loc_type')

    def add_x_noise(self):
        """Add input noise."""
        self.x += self.config['rng'].randn(*self.x.shape) * self._sigma_x


# non-timing tasks

def _spatial_reproduction(config, mode, **kwargs):
    '''
    A pulse with spatial information is presented, after a fixed delay comes another pulse, then the network should indicated the spatial location.
    '''
    dt = config['dt']
    rng = config['rng']
    pulse_duration = int(60/dt)
    response_duration = int(100/dt)

    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']
        prod_interval = (rng.uniform(100, 100, batch_size)/dt).astype(int)
        gaussian_center = rng.choice(np.arange(1., config['n_eachring']-1.), batch_size)
    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # the onset time of the first stimulus
    stim1_on = (rng.uniform(100, 100, batch_size)/dt).astype(int)

    # the offset time of the first stimulus
    stim1_off = stim1_on + pulse_duration
    # the onset time of the go cue
    control_on = stim1_off + prod_interval
    # the offset time of the go cue
    control_off = control_on + pulse_duration
    # response start time
    response_on = control_off
    # response end time
    response_off = response_on + response_duration
    xtdim = response_off

    trial = Trial(config, xtdim.max(), batch_size)
    # go cue
    #trial.add('input', 0, ons=control_on, offs=control_off, strengths=trial.expand(1.))
    # spatial input
    trial.add('line_gaussian_input', 1, ons=stim1_on, offs=stim1_off, strengths=trial.expand(1.), gaussian_center=gaussian_center)

    # output
    trial.add('line_gaussian_output', 0, ons=response_on, offs=response_off, strengths=trial.expand(1.), gaussian_center=gaussian_center)

    trial.cost_mask = np.zeros((xtdim.max(), batch_size, 1), dtype=trial.float_type)
    trial.add('cost_mask', 0, ons=response_on, offs=response_off, strengths=trial.expand(1.))

    trial.epochs = {'fix': (None, stim1_on),
                    'stim1': (stim1_on, stim1_off),
                    'interval': (stim1_off, control_on),
                    'go_cue': (control_on, control_off),
                    'go': (control_off, response_on),
                    'response': (response_on, response_off)}

    trial.prod_interval = prod_interval
    trial.gaussian_center = gaussian_center
    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()

    return trial


def spatial_reproduction(config, mode, **kwargs):
    return _spatial_reproduction(config, mode, **kwargs)


def _retro(config, mode, **kwargs):
    '''
    Two pulses with spatial locations are presented separately, with a fixed delay in-between. The
    network should indicate which pulse has larger location coordinate.
    '''
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ retro')

    dt = config['dt']
    rng = config['rng']
    stim_duration = int(config['stim'] / dt)
    stim_delay_interval = int(config['stim_delay'] / dt)
    cue_duration = int(config['cuestim'] / dt)
    cue_delay_interval = int(config['cue_delay'] / dt)
    response_duration = int(config['resp'] / dt)

    pref = np.arange(0, 2 * np.pi, 2 * np.pi / config['n_angle'])  # preferences
    # print('pref', pref * 180 / np.pi)



    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']
        var_value = 0*np.pi / 180
        var = rng.choice([var_value, -var_value], (batch_size,))

        stim_dist = rng.choice([1 * np.pi/2, -1 * np.pi/2], (batch_size,))
        gaussian_center1 = rng.choice(pref, (batch_size,))+var
        gaussian_center2 = (gaussian_center1 + stim_dist) % (2 * np.pi)+var
        cue_sign = rng.choice([1,-1], (batch_size,))


    elif mode == 'test':


        batch_size = kwargs['batch_size']

        gaussian_center1 = kwargs['gaussian_center1']
        if not hasattr(gaussian_center1, '__iter__'):
            gaussian_center1 = np.array([gaussian_center1] * batch_size)

        gaussian_center2 = kwargs['gaussian_center2']
        if not hasattr(gaussian_center2, '__iter__'):
            gaussian_center2 = np.array([gaussian_center2] * batch_size)

        cue_sign =  kwargs['cue_sign']
        if not hasattr(cue_sign, '__iter__'):
            cue_sign = np.array([cue_sign] * batch_size)

        # # print('***** cue_sign',cue_sign)
        # print('***** gaussian_center1',gaussian_center1)
        # print('***** gaussian_center2', gaussian_center2)



    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # the onset time of the first stimulus
    stim1_on = (rng.uniform(40, 40, batch_size) / dt).astype(int)
    stim1_off = stim1_on + stim_duration

    # the onset time of the second stimulus
    stim2_on = stim1_off + stim_delay_interval
    # the offset time of the second stimulus
    stim2_off = stim2_on + cue_duration
    # response start time
    response_on = stim2_off+cue_delay_interval
    # response end time
    response_off = response_on + response_duration
    xtdim = response_off
    #print('xtdim',xtdim)

    trial = Trial(config, xtdim.max(), batch_size)
    # print('stim1_on',stim1_on,stim1_off)
    # print('cue_on', cue_on,cue_off)

    # pulse input
    trial.add('line_gaussian_input', 0,                     ons=stim1_on, offs=stim1_off,strengths=trial.expand(config['in_strength']), gaussian_center=gaussian_center1)
    trial.add('line_gaussian_input', config['n_eachring'],  ons=stim1_on, offs=stim1_off,strengths=trial.expand(config['in_strength']), gaussian_center=gaussian_center2)
    trial.add('cue_input',           2*config['n_eachring'],ons=stim2_on, offs=stim2_off, strengths=config['in_strength']*cue_sign)
    trial.add('cue_input',           2*config['n_eachring']+1,ons=stim1_on, offs=response_off, strengths=trial.expand(config['in_strength']))


    # output
    gaussian_center_target = [gaussian_center1[i] if (cue_sign[i] == 1)
                 else gaussian_center2[i] for i in range(batch_size)]


    #print('gaussian_center_target', [ i* 180 / np.pi for i in np.array(gaussian_center_target)])
    trial.add('line_gaussian_output', 0, ons=response_on, offs=response_off, strengths=trial.expand(1),gaussian_center=gaussian_center_target)

    for i in range(config['n_eachring']):
        trial.add('cost_mask', i, ons=stim1_on, offs=response_off, strengths=trial.expand(config['cost_strength']))

    trial.epochs = {'fix': (stim1_on-2, stim1_on),
                    'stim1': (stim1_on, stim1_off),
                    'stim1_delay_interval': (stim1_off, stim2_on),
                    'stim2': (stim2_on, stim2_off),
                    'stim2_delay_interval': (stim2_off, response_on),
                    'response': (response_on, response_off)}

    #trial.cost_mask = np.zeros((xtdim.max(), batch_size, 1), dtype=trial.float_type)
    trial.gaussian_center1 = gaussian_center1
    trial.gaussian_center2 = gaussian_center2
    trial.gaussian_center_target = gaussian_center_target
    trial.cue_sign = cue_sign



    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()

    return trial
def _prosp(config, mode, **kwargs):
    '''
    Two pulses with spatial locations are presented separately, with a fixed delay in-between. The
    network should indicate which pulse has larger location coordinate.
    '''
    # print('+++++++++++++++++++++++++++++++++ prosp')
    dt = config['dt']
    rng = config['rng']
    stim_duration = int(config['stim'] / dt)
    stim_delay_interval = int(config['stim_delay'] / dt)
    cue_duration = int(config['cuestim'] / dt)
    cue_delay_interval = int(config['cue_delay'] / dt)
    response_duration = int(config['resp'] / dt)


    pref = np.arange(0, 2 * np.pi, 2 * np.pi / config['n_angle'])  # preferences

    if mode == 'random':  # Randomly generate parameters
        batch_size = kwargs['batch_size']
        var_value = 0 * np.pi / 180
        var = rng.choice([var_value, -var_value], (batch_size,))

        stim_dist = rng.choice([1 * np.pi/2, -1 * np.pi/2], (batch_size,))
        gaussian_center1 = rng.choice(pref, (batch_size,))+var
        gaussian_center2 = (gaussian_center1 + stim_dist) % (2 * np.pi)+var

        cue_sign = rng.choice([1,-1], (batch_size,))

        #print('gaussian_center1',gaussian_center1*180/np.pi)
        # print('gaussian_center2', gaussian_center2 * 180 / np.pi)
        # Target strengths
        #print('cue_sign',cue_sign)


    elif mode == 'test1':  # Randomly generate parameters
        batch_size = kwargs['batch_size']

        stim_dist = rng.choice([1 * np.pi, 1 * np.pi], (batch_size,))
        gaussian_center1 = rng.choice(pref, (batch_size,))
        gaussian_center2 = (gaussian_center1 + stim_dist) % (2 * np.pi)

        print('gaussian_center1',gaussian_center1*180/np.pi)
        print('gaussian_center2', gaussian_center2 * 180 / np.pi)
        # Target strengths

        #print('cue_sign',cue_sign)
        cue_sign = rng.choice([1,-1], (batch_size,))



    elif mode == 'test':
        batch_size = kwargs['batch_size']


        gaussian_center1 = kwargs['gaussian_center1']
        if not hasattr(gaussian_center1, '__iter__'):
            gaussian_center1 = np.array([gaussian_center1] * batch_size)

        gaussian_center2 = kwargs['gaussian_center2']
        if not hasattr(gaussian_center2, '__iter__'):
            gaussian_center2 = np.array([gaussian_center2] * batch_size)

        cue_sign = kwargs['cue_sign']
        if not hasattr(cue_sign, '__iter__'):
            cue_sign = np.array([cue_sign] * batch_size)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # the onset time of the first stimulus

    stim1_on = (rng.uniform(40, 40, batch_size) / dt).astype(int)
    stim1_off = stim1_on + cue_duration

    stim2_on = stim1_off + cue_delay_interval
    stim2_off = stim2_on + stim_duration
    response_on = stim2_off + stim_delay_interval
    response_off = response_on + response_duration

    xtdim = response_off
    #print('xtdim',xtdim)

    trial = Trial(config, xtdim.max(), batch_size)

    # pulse input
    trial.add('cue_input',           config['n_eachring']*2, ons=stim1_on,   offs=stim1_off,  strengths=config['in_strength']*cue_sign)
    trial.add('line_gaussian_input', 0,                    ons=stim2_on, offs=stim2_off,strengths=trial.expand(config['in_strength']), gaussian_center=gaussian_center1)
    trial.add('line_gaussian_input', config['n_eachring'], ons=stim2_on, offs=stim2_off,strengths=trial.expand(config['in_strength']), gaussian_center=gaussian_center2)

    trial.add('cue_input',           2*config['n_eachring']+2,ons=stim1_on, offs=response_off, strengths=trial.expand(config['in_strength']))

    # output
    gaussian_center_target = [gaussian_center1[i] if (cue_sign[i] == 1)
                 else gaussian_center2[i] for i in range(batch_size)]
    #print('gaussian_center_target', [ i* 180 / np.pi for i in np.array(gaussian_center_target)])
    trial.add('line_gaussian_output', 0, ons=response_on, offs=response_off,strengths=trial.expand(1),gaussian_center=gaussian_center_target)

    for i in range(config['n_eachring']):
        trial.add('cost_mask', i, ons=stim1_on, offs=response_off, strengths=trial.expand(config['cost_strength']))

    trial.epochs = {'fix': (stim1_on-2, stim1_on),
                    'stim1': (stim1_on, stim1_off),
                    'stim1_delay_interval': (stim1_off, stim2_on),
                    'stim2': (stim2_on, stim2_off),
                    'stim2_delay_interval': (stim2_off, response_on),

                    'response': (response_on, response_off)}

    #trial.cost_mask = np.zeros((xtdim.max(), batch_size, 1), dtype=trial.float_type)
    trial.gaussian_center1 = gaussian_center1
    trial.gaussian_center2 = gaussian_center2
    trial.gaussian_center_target = gaussian_center_target
    trial.cue_sign = cue_sign

    trial.seq_len = xtdim
    trial.max_seq_len = xtdim.max()
    return trial



def retro(config, mode, **kwargs):
    return _retro(config, mode, **kwargs)



def prosp(config, mode, **kwargs):
    return _prosp(config, mode, **kwargs)



def spatial_comparison2(config, mode, **kwargs):
    return _spatial_comparison(config, mode, **kwargs)


# map string to function
rule_mapping = {
                'spatial_reproduction': spatial_reproduction,
                'retro': retro,
                'prosp': prosp,

                }


def generate_trials(rule, hp, mode, noise_on=True, **kwargs):
    """Generate one batch of data.

    Args:
        hp: dictionary of hyperparameters
        mode: str, the mode of generating. Options: random, test, psychometric
        noise_on: bool, whether input noise is given

    Return:
        trial: Trial class instance, containing input and target output
    """
    # print('%%%%%rule%%%%rule%%%%%%rule%%%%rule%%%%%%',rule)
    config = hp
    kwargs['noise_on'] = noise_on
    trial = rule_mapping[rule](config, mode, **kwargs)
    trial.x=hp['input_scale']*trial.x
    trial.y = hp['input_scale']*trial.y

    if noise_on:
        trial.add_x_noise()

    return trial


# test the generation of trials
if __name__ == "__main__":

    from matplotlib import pyplot as plt
    import default

    rule_name = 'retro'
    hp = default.get_default_hp(rule_name)

    hp['n_eachring']=8

    hp['n_cue'] = 1
    hp['n_rule']=2
    hp['n_input'] = hp['n_eachring'] * 2 + hp['n_cue'] + hp['n_rule']
    hp['n_output'] = hp['n_eachring']

    hp['n_angle'] = 8

    hp['cue1']= 1
    hp['cue2'] = -1
    hp['sd_gaussianline'] = 8
    hp['cost_strength'] = 1
    hp['in_strength'] = 1

    hp['stim'] = 50
    hp['cuestim'] = 50
    hp['stim_delay']=50
    hp['cue_delay']=50
    hp['input_scale']=1
    hp['sigma_x']=0.0

    # print(hp['n_eachring'])
    train_time_interval = np.array([100, 100])
    batch_size=1

    trial = generate_trials(rule_name, hp, 'random', noise_on=True, batch_size=batch_size, train_time_interval=train_time_interval)
    # print(trial.epochs )
    gaussian_center1 = trial.gaussian_center1
    gaussian_center2 = trial.gaussian_center2


    length = trial.x.shape[0]
    start_x = int(0.0 * length)
    end_x = int(1 * length) + 1

    start = int(0.0 * length)
    end = int(1* length)

    for i in range(batch_size):
        print('==========================================================')
        print(gaussian_center1[i]*180/np.pi)
        print(gaussian_center2[i]*180/np.pi)


        print('trial_x', trial.x.shape, '\n', np.around(trial.x[start_x:end_x, i, :], decimals=1))
        print('trial_y', trial.y.shape, '\n', np.around(trial.y[start:end, i, :], decimals=1))
        #print('trial_cost_mask', trial.cost_mask.shape, '\n', np.around(trial.cost_mask[start:end, 0, :], decimals=1))


        if rule_name == 'retro':
            # input
            data_x_0 = trial.x[5, i, 0:hp['n_eachring']]
            data_x_1 = trial.x[5, i, hp['n_eachring']:2*hp['n_eachring']]
            data_y = trial.y[-1, i, :]
            fig, ax = plt.subplots()
            plt.plot(data_x_0, color='blue')
            plt.plot(data_x_1, color='red')
            #plt.plot(data_y, color='red')
            # plt.ylim([-0.1,1.5])
            #ax.axvline(x=(gaussian_center2*180/np.pi)/11.25,linewidth=3,color='g')

            plt.show()

            # # target output
            # plt.figure()
            # data = trial.y[:, 0, 0]
            # plt.plot(data, color='red')
            # plt.show()

        elif rule_name == 'prosp':
            # input
            data_x_0 = trial.x[0, i, 0:hp['n_eachring']]
            data_x_1 = trial.x[0, i, hp['n_eachring']:2 * hp['n_eachring']]
            data_y = trial.y[-1, i, :]
            fig, ax = plt.subplots()
            plt.plot(data_x_0, color='blue')
            plt.plot(data_x_1, color='red')
            # plt.plot(data_y, color='red')
            # plt.ylim([-0.1, 1.5])
            # ax.axvline(x=(gaussian_center2*180/np.pi)/11.25,linewidth=3,color='g')

            plt.show()

            # # target output
            # plt.figure()
            # data = trial.y[:, 0, 0]
            # plt.plot(data, color='red')
            # plt.show()

