import sys
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.nn.init as init
import pdb

import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import math
from scipy.stats import ortho_group
import tools



class EIRNN(nn.Module):
    """E-I RNN.
    Args:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
    Inputs:
        input: (seq_len, batch, input_size)
        hidden: (batch, hidden_size)
    """

    def __init__(self, hp, is_cuda=True):
        super().__init__()

        if is_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        hidden_size = hp['n_rnn']

        self.hp = hp
        self.hidden_size = hidden_size
        input_size = hp['n_input']
        output_size = hp['n_output']
        self.input_size = input_size


        self.n_md = int(hp['n_md'])
        self.n_rnn = int(hp['n_rnn'])
        self.e_size = int(self.n_rnn * hp['e_prop'])
        self.i_size = self.n_rnn - self.e_size


        n_ppc = int(hp['n_rnn'])
        n_pfc = int(hp['n_rnn'])

        mask = np.tile([1] * self.e_size + [-1] * self.i_size, (hidden_size, 1))
        np.fill_diagonal(mask, 0)
        self.mask = torch.tensor(mask, device=self.device, dtype=torch.float32)

        self.weight_ppc = nn.Parameter(torch.Tensor(n_ppc, n_ppc))
        init.kaiming_uniform_(self.weight_ppc, a=math.sqrt(5))
        self.weight_ppc.data[:, :self.e_size] /= self.n_rnn  # (self.n_rnn/self.i_size)

        self.weight_pfc = nn.Parameter(torch.Tensor(n_pfc, n_pfc))
        init.kaiming_uniform_(self.weight_pfc, a=math.sqrt(5))
        self.weight_pfc.data[:, :self.e_size] /= self.n_rnn  # (self.n_rnn/self.i_size)


        alpha_ppc = hp['dt'] / hp['tau_ppc']
        alpha_pfc = hp['dt'] / hp['tau_pfc']



        # self.dropout = nn.Dropout(p=0.1)


        #================= change the input =================
        #================= change the input =================
        weight_ih_ppc = torch.empty(input_size, hidden_size).uniform_(-1./math.sqrt(200.), 1./math.sqrt(200.))
        self.weight_ih_ppc = nn.Parameter(weight_ih_ppc)
        weight_ih_pfc = torch.empty(input_size, hidden_size).uniform_(-1. / math.sqrt(200.), 1. / math.sqrt(200.))
        self.weight_ih_pfc = nn.Parameter(weight_ih_pfc)
        #self.weight_ih = nn.Parameter(torch.Tensor(input_size, hidden_size))

        self.weight_out_ppc = nn.Parameter(torch.empty(n_ppc, output_size).normal_(0., 0.4/math.sqrt(hidden_size)))
        self.weight_out_pfc = nn.Parameter(torch.empty(n_pfc, output_size).normal_(0., 0.4/math.sqrt(hidden_size)))



        #self.weight_out = nn.Parameter(torch.Tensor(n_pfc, output_size))



        self.bias_out_ppc = nn.Parameter(torch.zeros(output_size,))
        self.bias_out_pfc = nn.Parameter(torch.zeros(output_size,))

        self.alpha_ppc = torch.tensor(alpha_ppc, device=self.device)
        self.sigma_rec_ppc = torch.tensor(math.sqrt(2./alpha_ppc) * hp['sigma_rec'], device=self.device)
        self.alpha_pfc = torch.tensor(alpha_pfc, device=self.device)
        self.sigma_rec_pfc = torch.tensor(math.sqrt(2. / alpha_pfc) * hp['sigma_rec'], device=self.device)

        self._0 = torch.tensor(0., device=self.device)
        self._1 = torch.tensor(1., device=self.device)

        act_fcn = hp['activation']
        if act_fcn == 'relu':
            self.act_fcn = lambda x: nn.functional.relu(x)
            #self.act_fcn = lambda x: F.relu(torch.clamp(x, -10, 10))
            #self.act_fcn = lambda x: nn.functional.leaky_relu(x, negative_slope=0.01)

        elif act_fcn == 'softplus':
            self.act_fcn = lambda x: nn.functional.softplus(x)

        elif act_fcn == 'tanh':
            self.act_fcn = lambda x: nn.functional.tanh(x)

        self.act_fcn_md = lambda x: nn.functional.sigmoid(x)

        self.weight_PPC_to_PFC = nn.Parameter(torch.Tensor(n_ppc, n_pfc))
        self.weight_PFC_to_PPC = nn.Parameter(torch.Tensor(n_pfc, n_ppc))

        # print('self.mask_cross_rnn:',self.mask_cross_rnn)


        self.PPC_to_MD_layer = torch.nn.Linear(n_ppc,self.n_md,  bias=False).to(self.device)
        self.PFC_to_MD_layer = torch.nn.Linear(n_pfc,self.n_md,  bias=False).to(self.device)

        self.MD_to_PPC_layer = torch.nn.Linear(self.n_md, n_ppc, bias=False).to(self.device)
        self.MD_to_PFC_layer = torch.nn.Linear(self.n_md,n_pfc,  bias=False).to(self.device)
        # self.MD_to_PPC_layer.weight.data.zero_()
        # self.MD_to_PFC_layer.weight.data.zero_()

        init.kaiming_uniform_(self.MD_to_PPC_layer.weight)
        init.kaiming_uniform_(self.MD_to_PFC_layer.weight)

        mask_input2ppc = np.tile([hp['alpha_input']*1] * self.e_size + [0] * (self.n_rnn - self.e_size), (input_size, 1))
        mask_input2pfc = np.tile([(1-hp['alpha_input'])*1] * self.e_size + [0] * (self.n_rnn - self.e_size), (input_size, 1))#np.tile([hp['sparsity_post']] * self.n_rnn, (input_size, 1))
        self.mask_input2ppc = torch.tensor(mask_input2ppc, device=self.device, dtype=torch.float32)
        self.mask_input2pfc = torch.tensor(mask_input2pfc, device=self.device, dtype=torch.float32)



        ######## ================================ sparsity connection ================================
        mask_cross_rnn = self.generate_custom_sparse_mask_ei(n_ppc, n_pfc, self.e_size, self.e_size,hp['sparsity_cross'])
        self.register_buffer('mask_cross_rnn', mask_cross_rnn)

        sparsity_MD_to_PPC = self.generate_custom_sparse_mask(self.n_md, n_ppc, self.hp['sparsity_md2ppc'])
        self.register_buffer('sparsity_MD_to_PPC', sparsity_MD_to_PPC)

        sparsity_MD_to_PFC = self.generate_custom_sparse_mask(self.n_md, n_pfc, self.hp['sparsity_md2pfc'])
        self.register_buffer('sparsity_MD_to_PFC', sparsity_MD_to_PFC)




        # hp['freeze'] = 'yes'
        # if hp['freeze']=='yes':
        #     for name, param in self.named_parameters():
        #         if name == 'MD_to_PPC_layer.weight':
        #             param.requires_grad = False
        #         if name == 'MD_to_PFC_layer.weight':
        #             param.requires_grad = False

        # self.init_block_connection(self.PPC_to_PFC_layer, num_blocks=2, scale=0.0)
        # self.init_block_connection(self.PFC_to_PPC_layer, num_blocks=2, scale=0.0)

    def generate_custom_sparse_mask_ei(self,n_ppc, n_pfc, n_exc_ppc, n_exc_pfc, sparsity_level):
        """
        Create a dense binary mask with specified sparsity between excitatory PPC and PFC units.

        Args:
            n_ppc (int): total number of PPC units
            n_pfc (int): total number of PFC units
            n_exc_ppc (int): number of excitatory PPC neurons
            n_exc_pfc (int): number of excitatory PFC neurons
            sparsity_level (float): proportion of zeros in E→E block (0.9 = 90% sparse)

        Returns:
            mask (torch.Tensor): [n_ppc, n_pfc] binary mask tensor
        """
        mask = torch.zeros(n_ppc, n_pfc)

        # Total entries in the E→E block
        total_ee = n_exc_ppc * n_exc_pfc
        num_nonzero = int((1 - sparsity_level) * total_ee)

        # Randomly sample which positions are active in E→E block
        idx_flat = torch.randperm(total_ee)[:num_nonzero]
        row_idx = idx_flat // n_exc_pfc
        col_idx = idx_flat % n_exc_pfc

        # Set selected entries to 1
        mask[row_idx, col_idx] = 1.0

        return mask

    def generate_custom_sparse_mask(self,n_ppc, n_pfc, sparsity_level):
        """
        Create a dense binary mask matrix between PPC and PFC with specified sparsity.

        Args:
            n_ppc (int): number of PPC units (rows)
            n_pfc (int): number of PFC units (columns)
            sparsity_level (float): proportion of zeros (e.g., 0.9 → 90% sparse)

        Returns:
            mask (torch.Tensor): [n_ppc, n_pfc] binary tensor (1 = connection exists)
        """
        total_connections = n_ppc * n_pfc
        num_active = int((1 - sparsity_level) * total_connections)

        # Randomly select which entries will be active
        flat_indices = torch.randperm(total_connections)[:num_active]
        row_idx = flat_indices // n_pfc
        col_idx = flat_indices % n_pfc

        # Initialize mask
        mask = torch.zeros(n_ppc, n_pfc)
        mask[row_idx, col_idx] = 1.0

        return mask


    def effective_weight_ppc(self):
        effective_weights = torch.abs(self.weight_ppc) * self.mask
        return effective_weights

    def effective_weight_pfc(self):
        effective_weights = torch.abs(self.weight_pfc) * self.mask
        return effective_weights


    def forward_rnn(self, inputs, init_state):
        """
        Propogate input through the network.
        inputs: torch.Size([time, batch_size, dim])
        ***inputs, init_state torch.Size([102, 100, 72]) torch.Size([100, 456])


        """
        #print('(rnn_ei1) inputs, init_state',inputs.shape, init_state.shape)
        #pdb.set_trace()

        # state = init_state
        # state_collector = [state]
        # print('init_state',init_state.shape)
        # sys.exit(0)

        state_PPC = 0*torch.rand((init_state.shape[0], init_state.shape[1])).to(self.device)
        state_PFC = 0*torch.rand((init_state.shape[0], init_state.shape[1])).to(self.device)
        state_MD  = 0*torch.rand((init_state.shape[0],  self.n_md)).to(self.device)



        state_collector_PPC = [state_PPC]
        state_collector_PFC = [state_PFC]
        state_collector_MD = [state_MD]
        t=0

        V_h1h2_list=[]
        V_h2h1_list=[]

        for input_per_step in inputs:

            #
            h1 = torch.matmul(self.act_fcn_md(state_MD), self.sparsity_MD_to_PPC * self.MD_to_PPC_layer.weight.T)
            h2 = torch.matmul(self.act_fcn_md(state_MD), self.sparsity_MD_to_PFC * self.MD_to_PFC_layer.weight.T)
            # print('state_MD:', state_MD[:3, 0])
            # print('h1:', h1[:3, 0])
            V_h1h1 = torch.mm(h1.T, h1)
            V_h2h2 = torch.mm(h2.T, h2)
            V_h1h2 = torch.mm(h1.T, h2)
            V_h2h1 = torch.mm(h2.T, h1)




            input_layer_ppc = torch.matmul(input_per_step, self.mask_input2ppc*self.weight_ih_ppc)
            input_layer_pfc = torch.matmul(input_per_step, self.mask_input2pfc*self.weight_ih_pfc)

            noise_ppc = torch.randn_like(state_PPC)*self.sigma_rec_ppc
            ##### PPC receive input from sensory_input, MD, PFC
            # print('==========self.PFC_to_PPC_layer.weight',self.PFC_to_PPC_layer.weight[:3,0])
            # print('state_MD:', state_MD[:3, 0])
            # print(t,'V_h1h2:', torch.mean(torch.abs(V_h1h2)).item())
            V_h1h2_list.append(torch.mean(torch.abs(V_h1h2)).item())
            V_h2h1_list.append(torch.mean(torch.abs(V_h2h1)).item())

            t += 1


            if self.hp['type'] == 'type0':
                PFC_to_PPC = torch.matmul(self.act_fcn(state_PFC), self.mask_cross_rnn*self.weight_PFC_to_PPC)#self.PFC_to_PPC_layer(self.act_fcn(state_PFC))  # self.PFC_to_PPC_layer(state_PFC)
                MD_to_PPC = torch.matmul(self.act_fcn_md(state_MD),self.sparsity_MD_to_PPC * self.MD_to_PPC_layer.weight.T)

                current_layer_PPC = F.linear(self.act_fcn(state_PPC), self.effective_weight_ppc())
                state_new_PPC     = input_layer_ppc + PFC_to_PPC + current_layer_PPC +MD_to_PPC+noise_ppc
            elif self.hp['type']=='type1':
                PFC_to_PPC = torch.matmul(self.act_fcn(state_PFC), self.hp['control_scale']*V_h2h1*self.mask_cross_rnn*self.weight_PFC_to_PPC)#self.PFC_to_PPC_layer(state_PFC)
                current_layer_PPC = F.linear(self.act_fcn(state_PPC), self.effective_weight_ppc())
                state_new_PPC     = input_layer_ppc + PFC_to_PPC + current_layer_PPC +noise_ppc




            state_PPC = (self._1 - self.alpha_ppc) * state_PPC + self.alpha_ppc * state_new_PPC
            ##### PFC receive inputs from PPC, MD
            noise_pfc = torch.randn_like(state_PFC) * self.sigma_rec_pfc


            if self.hp['type'] == 'type0':
                PPC_to_PFC = torch.matmul(self.act_fcn(state_PPC), self.mask_cross_rnn*self.weight_PPC_to_PFC)
                MD_to_PFC = torch.matmul(self.act_fcn_md(state_MD),self.sparsity_MD_to_PFC * self.MD_to_PFC_layer.weight.T)
                current_layer_PFC = F.linear(self.act_fcn(state_PFC), self.effective_weight_pfc())
                state_new_PFC     = input_layer_pfc +PPC_to_PFC + current_layer_PFC + MD_to_PFC + noise_pfc

            elif self.hp['type'] == 'type1':
                PPC_to_PFC = torch.matmul(self.act_fcn(state_PPC), self.hp['control_scale']*V_h1h2*self.mask_cross_rnn*self.weight_PPC_to_PFC)#self.PPC_to_PFC_layer(state_PPC)
                current_layer_PFC = F.linear(self.act_fcn(state_PFC), self.effective_weight_pfc())
                state_new_PFC     = input_layer_pfc +PPC_to_PFC + current_layer_PFC + noise_pfc


            state_PFC = (self._1 - self.alpha_pfc) * state_PFC + self.alpha_pfc * state_new_PFC

            ##### MD receive inputs from PPC, PFC
            PPC_to_MD = self.PPC_to_MD_layer(self.act_fcn(state_PPC))
            PFC_to_MD = self.PFC_to_MD_layer(self.act_fcn(state_PFC))

            # print('self.weight_ppc', self.weight_ppc[:10, 0])
            # print('self.weight_pfc', self.weight_pfc[:10, 0])
            # print('state_PPC',state_PPC[:10,0])
            # print('state_PFC', state_PFC[:10, 0])
            state_MD = PPC_to_MD + PFC_to_MD

            state_collector_MD.append(state_MD)

            # state_collector_MD.append(state_MD)
            state_collector_PPC.append(state_PPC)
            state_collector_PFC.append(state_PFC)



        # sys.exit(0)
        self.V_h1h2_list = V_h1h2_list
        self.V_h2h1_list = V_h2h1_list


        return state_collector_PPC,state_collector_PFC,state_collector_MD








    def out_weight_clipper(self):
        self.weight_out.data.clamp_(0.)

    def self_weight_clipper(self):
        diag_element_ppc = self.weight_ppc.diag().data.clamp_(0., 1.)
        self.weight_ppc.data[range(self.hidden_size), range(self.hidden_size)] = diag_element_ppc

        diag_element_pfc = self.weight_pfc.diag().data.clamp_(0., 1.)
        self.weight_pfc.data[range(self.hidden_size), range(self.hidden_size)] = diag_element_pfc


