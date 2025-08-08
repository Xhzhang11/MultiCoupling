import torch
import sys
import numpy as np
torch.set_printoptions(profile="full")
import pdb
from matplotlib import pyplot as plt

class TrainStepper(object):
    """The model"""

    """
    Initializing the model with information from hp

    Args:
        model_dir: string, directory of the hyper-parameters of the model
        hp: a dictionary or None
    """

    def __init__(self, model, hp, is_cuda=True):

        if is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        '''used during training for performance'''
        self._0 = torch.tensor(0., device=self.device)
        self._1 = torch.tensor(1., device=self.device)
        self._01 = torch.tensor(0.1, device=self.device)
        self._001 = torch.tensor(0.01, device=self.device)
        self._2 = torch.tensor(2., device=self.device)

        self.hp = hp
        self.n_ppc = int(hp['n_rnn'])
        self.n_pfc = int(hp['n_rnn'])


        # used hyper-parameters during training
        self.alpha = torch.tensor(hp['alpha'], device=self.device)

        self.net = model

        if is_cuda:
            self.net.cuda(device=self.device)

        # weight list, used for regularization

        self.weight_list = [self.net.RNN_layer.weight_ih_ppc, self.net.RNN_layer.weight_ih_pfc,self.net.RNN_layer.weight_ppc,
                            self.net.RNN_layer.weight_pfc,self.net.RNN_layer.weight_PPC_to_PFC,
                            self.net.RNN_layer.weight_PFC_to_PPC,
                            self.net.RNN_layer.weight_out_ppc,
                            self.net.RNN_layer.weight_out_pfc]
        self.out_weight_ppc = self.net.RNN_layer.weight_out_ppc
        self.out_weight_pfc = self.net.RNN_layer.weight_out_pfc
        # print('self.out_weight:',self.out_weight.shape)



        self.out_bias_ppc = self.net.RNN_layer.bias_out_ppc
        self.out_bias_pfc = self.net.RNN_layer.bias_out_pfc
        self.act_fcn = self.net.RNN_layer.act_fcn

        # regularization parameters
        self.l1_weight = torch.tensor(hp['l1_weight'], device=self.device)
        self.l2_weight = torch.tensor(hp['l2_weight'], device=self.device)

        self.l2_firing_rate = torch.tensor(hp['l2_firing_rate'], device=self.device)
        self.l1_firing_rate = torch.tensor(hp['l1_firing_rate'], device=self.device)

        self.l1_weight_cpu = torch.tensor(hp['l1_weight'], device=torch.device("cpu"))
        self.l2_weight_cpu = torch.tensor(hp['l2_weight'], device=torch.device("cpu"))

        self.l2_firing_rate_cpu = torch.tensor(hp['l2_firing_rate'], device=torch.device("cpu"))
        self.l1_firing_rate_cpu = torch.tensor(hp['l1_firing_rate'], device=torch.device("cpu"))

        if hp['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=hp['learning_rate'])
        elif hp['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=hp['learning_rate'])

    def forward(self, inputs, initial_state):
        return self.net.forward(inputs, initial_state)

    def cost_fcn(self, **kwargs):
        '''
        :param inputs: GPU tensor (time, batch_size, input_size)
        :param target_outputs: GPU tensor (time, batch_size, output_size)
        :param cost_mask: GPU tensor
        :param relu_mask: GPU tensor
        :param cost_start_time: CPU int
        :param cost_end_time: CPU int
        :param initial_state: GPU tensor
        :return:
        '''

        inputs = kwargs['inputs'].to(self.device)

        target_outputs = kwargs['target_outputs'].to(self.device)

        # print('target_outputs',target_outputs.shape)
        # sys.exit(0)
        cost_mask = kwargs['cost_mask'].to(self.device)
        cost_start_time = kwargs['cost_start_time']
        cost_end_time = kwargs['cost_end_time']
        initial_state = kwargs['initial_state'].to(self.device)
        # shape(time, batch_size, hidden_size=1)
        seq_mask = kwargs['seq_mask'].type(torch.float32).unsqueeze(2)
        seq_mask = seq_mask.to(self.device)




        batch_size, hidden_size = initial_state.shape

        #print('(train_stepper) inputs, initial_state',inputs.shape, initial_state.shape)

        state_collector_PPC,state_collector_PFC,state_collector_MD = self.forward(inputs, initial_state)



        # calculate cost_lsq
        # cost_end_time = torch.max(seq_len).item()
        # batch_size = inputs.shape[1]

        # shape(time, batch_size, hidden_size)
        # need to +1 here, because state_collector also collect the initial state
        #print(cost_start_time,cost_end_time)

        state_binder_PPC = torch.cat(state_collector_PPC[cost_start_time + 1 :cost_end_time + 1 ], dim=0).view(-1, batch_size, hidden_size)
        state_binder_PFC = torch.cat(state_collector_PFC[cost_start_time + 1 :cost_end_time + 1 ], dim=0).view(-1, batch_size, hidden_size)
        state_binder_MD  = torch.cat(state_collector_MD[cost_start_time + 1 :cost_end_time + 1 ], dim=0).view(-1, batch_size,self.hp['n_md'])
        # print('state_binder_PPC=', state_binder_PPC.shape)
        # print('state_binder_PFC=',state_binder_PFC.shape)
        state_binder = torch.cat([state_binder_PPC, 1*state_binder_PFC], dim=-1)  # shape: [48, 256, 400]

        self.firing_rate_binder_parietal = self.act_fcn(state_binder_PPC)
        self.firing_rate_binder_pfc = self.act_fcn(state_binder_PFC)
        self.firing_rate_binder_MD = self.act_fcn(state_binder_MD)
        self.firing_rate_binder = self.act_fcn(state_binder)

        self.V_h1h2_list = self.net.RNN_layer.V_h1h2_list
        self.V_h2h1_list = self.net.RNN_layer.V_h2h1_list

        # print('self.firing_rate_binder_ppc',self.firing_rate_binder_ppc.shape)




        #pdb.set_trace()

        #self.net.dropout_layer(self.firing_rate_binder)


        self.outputs_ppc = torch.matmul(self.firing_rate_binder_parietal, self.out_weight_ppc) + self.out_bias_ppc
        self.outputs_pfc = torch.matmul(self.firing_rate_binder_pfc, self.out_weight_pfc) + self.out_bias_pfc
        self.outputs = self.hp['alpha_output']*self.outputs_ppc+(1-self.hp['alpha_output'])*self.outputs_pfc



        # add dropout layer

        cost_mask_length = torch.sum(cost_mask, dim=0)
        #print('cost_mask_length',cost_mask_length.shape)

        self.cost_lsq = torch.mean(torch.sum(((self.outputs - target_outputs) ** self._2), dim=0))

        #self.cost_lsq = torch.mean(torch.sum(((self.outputs - target_outputs) ** self._2) * cost_mask, dim=0) / cost_mask_length)

        # calculate cost_reg
        self.cost_reg = self._0
        if self.l1_weight_cpu > 0:
            temp = self._0
            for x in self.weight_list:
                temp = temp + torch.mean(torch.abs(x))
            self.cost_reg = self.cost_reg + temp * self.l1_weight

        if self.l2_weight_cpu > 0:
            temp = self._0
            for x in self.weight_list:
                temp = temp + torch.mean(x ** self._2)
            self.cost_reg = self.cost_reg + temp * self.l2_weight

        if self.l2_firing_rate_cpu > 0:
            seq_mask_n_element = torch.sum(seq_mask, dim=0)
            self.cost_reg = self.cost_reg + torch.mean(torch.sum((self.firing_rate_binder_parietal * seq_mask) ** self._2,
                                                                 dim=0) / seq_mask_n_element) * self.l2_firing_rate\
                            + torch.mean(torch.sum((self.firing_rate_binder_pfc * seq_mask) ** self._2,
                                                                 dim=0) / seq_mask_n_element) * self.l2_firing_rate


        if self.l1_firing_rate_cpu > 0:
            seq_mask_n_element = torch.sum(seq_mask, dim=0)
            self.cost_reg = self.cost_reg + torch.mean(torch.sum(torch.abs(self.firing_rate_binder_parietal * seq_mask),
                                                                 dim=0) / seq_mask_n_element) * self.l1_firing_rate\
                            + torch.mean(torch.sum(torch.abs(self.firing_rate_binder_pfc * seq_mask),
                                                                 dim=0) / seq_mask_n_element) * self.l1_firing_rate
        #print('self.cost_reg',self.cost_reg)
        self.cost = self.cost_lsq + self.cost_reg
        #sys.exit(0)

    def stepper(self, **kwargs):

        self.optimizer.zero_grad()

        self.cost_fcn(**kwargs)
        self.cost.backward()

        if self.cost > 0.1:
            torch.nn.utils.clip_grad_value_(self.net.parameters(), self._1)
        else:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self._1, 2)

        self.optimizer.step()

        self.net.RNN_layer.self_weight_clipper()
