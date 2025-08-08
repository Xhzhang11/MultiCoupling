import torch
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

        # used hyper-parameters during training
        self.alpha = torch.tensor(hp['alpha'], device=self.device)
        self.net = model

        if is_cuda:
            self.net.cuda(device=self.device)

        # weight list, used for regularization

        if hp['net']=='rnn':
            self.weight_list = [self.net.input2h.weight, self.net.weight_hh, self.net.weight_out]
        elif hp['net'] == 'pfcmd':
            self.weight_list = [self.net.input2h.weight, self.net.weight_hh, self.net.weight_out,

                                self.net.linear_layer_MDtoPFC.weight, self.net.linear_layer_PFCtoMD.weight]


            # self.weight_list = [
            #                     self.net.weight_MDtoPFC, self.net.linear_layer_MDtoPFC.weight,
            #                    ]

            # self.weight_list = [self.net.input2h.weight, self.net.weight_hh, self.net.weight_out,
            #                     self.net.linear_layer_cuetoMD.weight,
            #                     self.net.weight_MDtoPFC,
            #                     self.net.linear_layer_PFCtoMD.weight]

        self.out_weight = self.net.weight_out
        # print('self.out_weight:',self.out_weight.shape)


        self.hidden_weight = self.net.weight_hh

        self.out_bias = self.net.bias_out
        self.act_fcn = self.net.act_fcn
        self.act_fcn_md = self.net.act_fcn_md


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

        self.cost_lsq = self._0
        self.cost_reg = self._0

    def forward(self, inputs, initial_state,initial_state_md):
        state_collector, state_collector_MD, Vt_list = self.net.forward(inputs, initial_state,initial_state_md)
        return state_collector, state_collector_MD, Vt_list

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
        # print('inputs',inputs.shape)

        target_outputs = kwargs['target_outputs'].to(self.device)
        cost_mask = kwargs['cost_mask'].to(self.device)
        cost_start_time = kwargs['cost_start_time']
        cost_end_time = kwargs['cost_end_time']
        initial_state = kwargs['initial_state'].to(self.device)
        initial_state_md = kwargs['initial_state_md'].to(self.device)
        # shape(time, batch_size, hidden_size=1)
        seq_mask = kwargs['seq_mask'].type(torch.float32).unsqueeze(2)
        seq_mask = seq_mask.to(self.device)
        # print(seq_mask)

        batch_size, hidden_size = initial_state.shape
        n_md = self.net.n_md


        #print('(train_stepper) inputs, initial_state',inputs.shape, initial_state.shape)

        state_collector,state_collector_md,V_t = self.forward(inputs, initial_state,initial_state_md)
        self.V_t = V_t
        #pdb.set_trace()

        # calculate cost_lsq
        # cost_end_time = torch.max(seq_len).item()
        # batch_size = inputs.shape[1]

        # shape(time, batch_size, hidden_size)
        # need to +1 here, because state_collector also collect the initial state
        #print(cost_start_time,cost_end_time)

        state_binder = torch.cat(state_collector[cost_start_time + 1 :cost_end_time + 1 ], dim=0).view(-1, batch_size, hidden_size)
        state_binder_md = torch.cat(state_collector_md[cost_start_time+ 1:cost_end_time+ 1], dim=0).view(-1, batch_size, n_md)


        self.firing_rate_binder = self.act_fcn(state_binder)

        self.firing_rate_md = self.act_fcn_md(state_binder_md)

        #pdb.set_trace()

        #self.net.dropout_layer(self.firing_rate_binder)

        self.outputs = torch.matmul(self.firing_rate_binder, self.out_weight) + self.out_bias
        # add dropout layer



        cost_mask_length = torch.sum(cost_mask, dim=0)

        #self.cost_lsq = torch.mean(torch.sum(((self.outputs - target_outputs) ** self._2) * cost_mask, dim=0) / cost_mask_length)
        cost_lsq_1 = torch.mean(torch.sum(((self.outputs - target_outputs) ** self._2) * cost_mask, dim=0) / cost_mask_length)
        cost_lsq_2 = torch.mean(torch.sum(((self.outputs - target_outputs) ** self._2), dim=0))
        self.cost_lsq = self.hp['lsq1'] * cost_lsq_1 + self.hp['lsq2'] * cost_lsq_2

        # calculate cost_reg

        self.cost_reg = self._0

        # if self.l1_weight_cpu > 0:
        #     temp = self._0
        #     for x in self.weight_list:
        #         temp = temp + torch.mean(torch.abs(x))
        #         #print('********* temp',temp)
        #     self.cost_reg = self.cost_reg + temp * self.l1_weight
        #
        #
        # if self.l2_weight_cpu > 0:
        #     temp = self._0
        #     for x in self.weight_list:
        #         temp = temp + torch.mean(x ** self._2)
        #     self.cost_reg = self.cost_reg + temp * self.l2_weight
        #
        #
        # if self.l2_firing_rate_cpu > 0:
        #     #print('self.l2_firing_rate_cpu',self.l2_firing_rate_cpu)
        #     seq_mask_n_element = torch.sum(seq_mask, dim=0)
        #     self.cost_reg = self.cost_reg + torch.mean(torch.sum((self.firing_rate_binder * seq_mask) ** self._2,
        #                                                          dim=0) / seq_mask_n_element) * self.l2_firing_rate
        #
        #     self.cost_reg = self.cost_reg + torch.mean(torch.sum((self.firing_rate_md * seq_mask) ** self._2,
        #                                                          dim=0) / seq_mask_n_element) * self.l2_firing_rate
        #
        # if self.l1_firing_rate_cpu > 0:
        #     seq_mask_n_element = torch.sum(seq_mask, dim=0)
        #     self.cost_reg = self.cost_reg + torch.mean(torch.sum(torch.abs(self.firing_rate_binder * seq_mask),
        #                                                          dim=0) / seq_mask_n_element) * self.l1_firing_rate
        #     self.cost_reg = self.cost_reg + torch.mean(torch.sum(torch.abs(self.firing_rate_md * seq_mask),
        #                                                          dim=0) / seq_mask_n_element) * self.l1_firing_rate
        #

        # print('##################################################self.cost_reg_md',self.cost_reg_md)
        self.cost = self.cost_lsq + self.cost_reg
        #print('V_list',len(V_list),V_list[0].shape)



    def stepper(self, **kwargs):

        self.optimizer.zero_grad()

        self.cost_fcn(**kwargs)
        self.cost.backward()

        # if self.cost > 0.1:
        #     torch.nn.utils.clip_grad_value_(self.net.parameters(), self._1)
        # else:
        #     torch.nn.utils.clip_grad_norm_(self.net.parameters(), self._1, 2)

        self.optimizer.step()

        self.net.self_weight_clipper()
        self.net.out_weight_clipper()


