# -*- coding: utf-8 -*-
import os
import sys

import torch
from torch import nn
import tools
import torch.nn.init as init
import pdb
from torch.nn import functional as F
import numpy as np
import math




class NetworkFeedForward(nn.Module):
    def __init__(self, hp, is_cuda=True):
        super(NetworkFeedForward, self).__init__()

        if is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.hp = hp
          # rnn
        input_size = hp['n_input']
        hidden_size = hp['n_rnn']
        output_size = hp['n_output']
        rnn_exc_inh = hp['n_rnn']

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.n_md = int(hp['n_md'])
        self.n_rnn = int(hp['n_rnn'])

        alpha = hp['dt'] / hp['tau']
        self.alpha = alpha
        self.sigma_rec = hp['sigma_rec']


        act_fcn = hp['activation']
        act_fcn_md = hp['activation_md']
        if act_fcn == 'relu':
            self.act_fcn = lambda x: nn.functional.relu(x)
        elif act_fcn == 'softplus':
            self.act_fcn = lambda x: nn.functional.softplus(x)
        elif act_fcn == 'sig':
            self.act_fcn = lambda x: nn.functional.sigmoid(x)
        elif act_fcn == 'tanh':
            self.act_fcn = lambda x: nn.functional.tanh(x)

        if act_fcn_md == 'relu':
            self.act_fcn_md = lambda x: nn.functional.relu(x)
        elif act_fcn_md == 'softplus':
            self.act_fcn_md = lambda x: nn.functional.softplus(x)
        elif act_fcn_md == 'sig':
            self.act_fcn_md = lambda x: nn.functional.sigmoid(x)
        elif act_fcn_md == 'tanh':
            self.act_fcn_md = lambda x: nn.functional.tanh(x)



        self.input2h = nn.Linear(input_size, hidden_size)

        #################### recurrent #########################################
        self.bias = nn.Parameter(torch.Tensor(hidden_size))

        if hp['initial_hh']=='zero':
            self.weight_hh = nn.Parameter(0 * torch.empty(hidden_size, hidden_size).normal_(0., 0.0 / math.sqrt(hidden_size)))
        elif hp['initial_hh']=='Xavier':
            self.weight_hh = nn.Parameter(hp['scale_init']*torch.Tensor(hidden_size, hidden_size))
        elif hp['initial_hh'] == 'kaiming':
            self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            init.kaiming_uniform_(self.weight_hh, a=0)

        elif hp['initial_hh'] == 'uniform':
            self.weight_hh = nn.Parameter(hp['scale_init']*torch.empty(hidden_size, hidden_size).uniform_(-1. / math.sqrt(2.), 1. / math.sqrt(2.)))



        ################## output#########################################
        self.weight_out = nn.Parameter(torch.empty(hidden_size, output_size).normal_(0., 0.4 / math.sqrt(hidden_size)))
        self.bias_out = nn.Parameter(torch.zeros(output_size, ))

        self.alpha = torch.tensor(alpha, device=self.device)
        self.sigma_rec = torch.tensor(math.sqrt(2. / alpha) * hp['sigma_rec'], device=self.device)

        self._0 = torch.tensor(0., device=self.device)
        self._1 = torch.tensor(1., device=self.device)

        #self.linear_layer_cuetoMD = nn.Linear(self.input_size, self.n_md,)
        self.linear_layer_MDtoPFC = nn.Linear(self.n_md, self.hidden_size,bias=False)
        self.linear_layer_PFCtoMD = nn.Linear(self.hidden_size, self.n_md,bias=False)

        if self.hp['freeze']=='freeze':
            self._initialize_weights()

            for name, param in self.named_parameters():
                if name == 'linear_layer_MDtoPFC.weight':
                    param.requires_grad = False
                if name == 'linear_layer_PFCtoMD.weight':
                    param.requires_grad = False



    def generate_sparsity_matrix_fixed_ratio(self, rows, cols, zero_ratio):
        # Total number of elements
        total_elements = rows * cols

        # Calculate the number of zeros
        num_zeros = int(total_elements * zero_ratio)

        if num_zeros > total_elements:
            raise ValueError("The ratio of zeros cannot exceed 1.0 (100%).")

        # Create a flat tensor with the desired number of zeros and ones
        flat_matrix = torch.cat([
            torch.zeros(num_zeros, dtype=torch.int),
            torch.ones(total_elements - num_zeros, dtype=torch.int)
        ])

        # Shuffle the tensor to randomize the positions of zeros and ones
        shuffled_matrix = flat_matrix[torch.randperm(total_elements)]

        # Reshape into the desired dimensions
        sparsity_matrix = shuffled_matrix.view(rows, cols)

        return sparsity_matrix

    def _initialize_weights(self):
        sparsity_level_MDtoPFC = self.hp['sparsity_weight']#0.8
        sparsity_level_PFCtoMD = self.hp['sparsity_weight']#0.8

        sparsity_MDtoPFC = self.generate_sparsity_matrix_fixed_ratio(self.hidden_size,self.n_md, sparsity_level_MDtoPFC)#torch.bernoulli(torch.full((self.hidden_size,self.n_md), 1 - sparsity_level_MDtoPFC)).int()
        sparsity_PFCtoMD = self.generate_sparsity_matrix_fixed_ratio(self.n_md,self.hidden_size, sparsity_level_PFCtoMD)

        MDtoPFC_weight = sparsity_MDtoPFC*torch.empty(self.hidden_size,self.n_md).uniform_(-1. / math.sqrt(100.), 1. / math.sqrt(100.))
        PFCtoMD_weight = sparsity_PFCtoMD*torch.empty(self.n_md,self.hidden_size).uniform_(-1. / math.sqrt(100.), 1. / math.sqrt(100.))




        self.linear_layer_MDtoPFC.weight = nn.Parameter(MDtoPFC_weight)
        self.linear_layer_PFCtoMD.weight = nn.Parameter(PFCtoMD_weight)


        # if self.linear_layer_MDtoPFC.bias is not None:
        #     init.constant_(self.linear_layer_MDtoPFC.bias, 0)




    def effective_weight(self):
        #effective_weights = torch.abs(self.weight_hh) * self.mask
        effective_weights = self.weight_hh
        return effective_weights


    def generate_low_rank(self):


        # Set the dimensions of the matrix
        rows = self.n_rnn
        cols = self.n_rnn
        rank = self.hp['rank']

        # Generate two random vectors
        u = np.random.rand(rows, rank).astype(np.float32)
        v = np.random.rand(rank, cols).astype(np.float32)

        # Create the low-rank matrix
        low_rank_matrix = np.dot(u, v)
        low_rank_matrix =self.hp['scale_value']*torch.tensor(low_rank_matrix, device=self.device)
        #print('low_rank_matrix',low_rank_matrix.shape)

        # print("Low-rank matrix:")
        # print(low_rank_matrix)
        # print("Rank of the matrix:", np.linalg.matrix_rank(low_rank_matrix))

        return low_rank_matrix


    def generate_random_rank(self):
        random_matrix = np.random.rand(self.hidden_size, self.hidden_size).astype(np.float32)  # Replace hidden_size as needed

        # Generate the matrix with values in the range [-1, 1]

        random_matrix = self.hp['scale_value']*torch.tensor(random_matrix, device=self.device)

        return random_matrix

    def generate_ones(self):

        # Generate a 3x3 matrix filled with 1s
        matrix = np.ones((self.hidden_size, self.hidden_size), dtype=np.float32)
        matrix_ones = self.hp['scale_value'] * torch.tensor(matrix, device=self.device)

        return matrix_ones

    def forward(self, inputs, init_state,initial_state_md):
        """
        Propogate input through the network.
        inputs: torch.Size([time, batch_size, dim])
        ***inputs, init_state torch.Size([102, 100, 72]) torch.Size([100, 456])
        """
        #print('(rnn_ei1) inputs, init_state',inputs.shape, init_state.shape)
        #pdb.set_trace()
        state = init_state

        state_collector = [state]
        state_collector_MD = [initial_state_md]
        Vt_list = []
        #print(self.effective_weight()[0,:10])

        # Loop through time
        for input_per_step in inputs:
            if self.hp['type']=='type0':

                # print('state',state.shape)

                PFCtoMD_layer_input = self.linear_layer_PFCtoMD(state)
                MD_receive = PFCtoMD_layer_input
                MD_state = self.act_fcn_md(MD_receive)
                ######## PFC receive ########
                MD2pfc_layer_input = self.linear_layer_MDtoPFC(MD_state)  # torch.matmul(MD_state, self.weight_MDtoPFC)


                h = MD2pfc_layer_input
                V_t = self.effective_weight()
                # matrix_V = V_t.detach().cpu().numpy()
                # rank = np.linalg.matrix_rank(matrix_V)
                # print("Rank of the tensor:", rank)

                input_layer = self.input2h(input_per_step)
                current_layer = F.linear(state, self.effective_weight(), self.bias)
                noise = torch.randn_like(state) * self.sigma_rec

                state_new = self.act_fcn(current_layer) + input_layer + MD2pfc_layer_input + noise
                state = (self._1 - self.alpha) * state + self.alpha * state_new

                state_collector.append(state)
                state_collector_MD.append(MD_state)


            elif self.hp['type']=='type1':

                PFCtoMD_layer_input = self.linear_layer_PFCtoMD(state)
                MD_receive =  PFCtoMD_layer_input
                MD_state = self.act_fcn_md(MD_receive)
                ######## PFC receive ########
                r_MD = MD_state#[0,:][np.newaxis,:]
                #print('r_MD',r_MD.shape,self.linear_layer_MDtoPFC.weight.T.shape)
                h = torch.matmul(r_MD, self.linear_layer_MDtoPFC.weight.T)
                # print('h',h.shape)

                V_t = self.hp['scale_value']*torch.mm(h.T, h)  # 256*256


                input_layer = self.input2h(input_per_step)
                current_layer = F.linear(state, torch.mul(V_t,self.effective_weight()), self.bias)
                noise = torch.randn_like(state) * self.sigma_rec
                state_new = self.act_fcn(current_layer) + input_layer + noise
                state = (self._1 - self.alpha) * state + self.alpha * state_new

                state_collector.append(state)
                state_collector_MD.append(MD_state)
            #print('V: ',  V_t[:5, 0].data)
            if self.hp['type']=='type2':
                PFCtoMD_layer_input = self.linear_layer_PFCtoMD(state)
                MD_receive = PFCtoMD_layer_input
                MD_state = self.act_fcn_md(MD_receive)
                ######## PFC receive ########
                MD2pfc_layer_input = self.linear_layer_MDtoPFC(MD_state)  # torch.matmul(MD_state, self.weight_MDtoPFC)

                V_t = self.generate_low_rank()
                # matrix_V = V_t.detach().cpu().numpy()
                # rank = np.linalg.matrix_rank(matrix_V)
                # print("Rank of the tensor:", rank)

                input_layer = self.input2h(input_per_step)
                current_layer = F.linear(state, torch.mul(V_t, self.effective_weight()), self.bias)
                noise = torch.randn_like(state) * self.sigma_rec

                state_new = self.act_fcn(current_layer) + input_layer + MD2pfc_layer_input + noise
                state = (self._1 - self.alpha) * state + self.alpha * state_new

                state_collector.append(state)
                state_collector_MD.append(MD_state)


            if self.hp['type']=='type3':
                PFCtoMD_layer_input = self.linear_layer_PFCtoMD(state)
                MD_receive = PFCtoMD_layer_input
                MD_state = self.act_fcn_md(MD_receive)
                ######## PFC receive ########
                MD2pfc_layer_input = self.linear_layer_MDtoPFC(MD_state)  # torch.matmul(MD_state, self.weight_MDtoPFC)

                V_t = self.generate_random_rank()
                # matrix_V = V_t.detach().cpu().numpy()
                # rank = np.linalg.matrix_rank(matrix_V)
                # print("Rank of the tensor:", rank)

                input_layer = self.input2h(input_per_step)
                current_layer = F.linear(state, torch.mul(V_t, self.effective_weight()), self.bias)
                noise = torch.randn_like(state) * self.sigma_rec

                state_new = self.act_fcn(current_layer) + input_layer + MD2pfc_layer_input + noise
                state = (self._1 - self.alpha) * state + self.alpha * state_new

                state_collector.append(state)
                state_collector_MD.append(MD_state)

            if self.hp['type']=='type4':
                PFCtoMD_layer_input = self.linear_layer_PFCtoMD(state)
                MD_receive = PFCtoMD_layer_input
                MD_state = self.act_fcn_md(MD_receive)
                ######## PFC receive ########
                MD2pfc_layer_input = self.linear_layer_MDtoPFC(MD_state)  # torch.matmul(MD_state, self.weight_MDtoPFC)

                V_t = self.generate_ones()
                # matrix_V = V_t.detach().cpu().numpy()
                # rank = np.linalg.matrix_rank(matrix_V)
                # print("Rank of the tensor:", rank)

                input_layer = self.input2h(input_per_step)
                current_layer = F.linear(state, torch.mul(V_t, self.effective_weight()), self.bias)
                noise = torch.randn_like(state) * self.sigma_rec

                state_new = self.act_fcn(current_layer) + input_layer + MD2pfc_layer_input + noise
                state = (self._1 - self.alpha) * state + self.alpha * state_new

                state_collector.append(state)
                state_collector_MD.append(MD_state)

            Vt_list.append(h)


        # print('V_t',V_t[0:20,0:20])
        # matrix = self.effective_weight().detach().cpu().numpy()
        # rank = np.linalg.matrix_rank(matrix)
        # print("Rank of effective_weight:", rank)



        # new_matrix = torch.mul(V_t, self.effective_weight())
        # rank = np.linalg.matrix_rank(new_matrix)
        # print("Rank of the tensor:", rank)




        return state_collector,state_collector_MD,Vt_list


    def out_weight_clipper(self):
        self.weight_out.data.clamp_(0.)

    def self_weight_clipper(self):
        diag_element = self.weight_hh.diag().data.clamp_(0., 1.)
        self.weight_hh.data[range(self.hidden_size), range(self.hidden_size)] = diag_element


    def save(self,model_dir):
        save_path = os.path.join(model_dir, 'most_recent_model.pth')
        torch.save(self.state_dict(), save_path)


    def load(self, model_dir):
        if model_dir is not None:
            save_path = os.path.join(model_dir, 'most_recent_model.pth')
            if os.path.isfile(save_path):
                self.load_state_dict(torch.load(save_path, map_location=lambda storage, loc: storage), strict=False)






