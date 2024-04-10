'''
Citation: https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py

'''

import torch
from torch import nn
from torch.nn import functional as F
from functools import reduce
from torch.autograd import Variable

from .utils import Encoder, _extend_for_multilayer, _update_basic_layer, _make_aux_layers

from typing import Tuple, List, Dict, NewType, Callable
Tensor = NewType('Tensor', torch.tensor)

class BasicLSTMCell(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int) -> None:

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gate = nn.Identity()
        self.drop_out = nn.Identity()

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([self.drop_out(input_tensor), h_cur], dim=1)  # concatenate along channel axis

        combined_gate = self.gate(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_gate, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return [h_next, c_next]

    def init_hidden(self, batch_size, image_size):
        hidden_sizes = tuple([batch_size, self.hidden_dim] + image_size)
        return (Variable(torch.zeros(hidden_sizes, device=self.gate.weight.device)),
                Variable(torch.zeros(hidden_sizes, device=self.gate.weight.device)))

class LSTMCell(BasicLSTMCell):

    def __init__(self, input_dim: int, hidden_dim: int, drop_out: float = 0.2, bias: bool = True):

        super().__init__(input_dim, hidden_dim)

        self.gate = nn.Linear(in_features=self.input_dim + self.hidden_dim,
                              out_features=4 * self.hidden_dim,
                              bias=bias)
        if drop_out > 0.:
            self.drop_out = nn.Dropout(p=drop_out)

class ConvLSTMCell(BasicLSTMCell):

    def __init__(self, input_dim: int,
                       hidden_dim: int,
                       kernel_size: int = 3, 
                       stride: int = 1, 
                       padding: int = 1, 
                       bias: bool = True):
        """
        Initialize ConvLSTM cell. (for 1D conv)

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super().__init__(input_dim, hidden_dim)

        self.gate = nn.Conv1d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)

class BiConvLSTMCell(ConvLSTMCell):

    '''
    
    
    '''
    def __init__(self, input_dim: int, hidden_dim: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 1, 
                 kernel_size_cat: int = 1, stride_cat: int = 1, padding_cat: int = 1,
                 bias=True):
        super().__init__(input_dim, hidden_dim, kernel_size, stride, padding, bias)

        self.concat = nn.Conv1d(in_channels=2 * hidden_dim,
                                out_channels=hidden_dim,
                                kernel_size=kernel_size_cat,
                                stride=stride_cat,
                                padding=padding_cat,
                                bias=bias)

class BasicGRUCell(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv_gates = nn.Identity()  # for update_gate,reset_gate respectively
        self.conv_can  = nn.Identity()  # for candidate neural memory 
        self.drop_out = nn.Identity()

    def init_hidden(self, batch_size, image_size):
        hidden_sizes = tuple([batch_size, self.hidden_dim] + image_size)
        return [Variable(torch.zeros(hidden_sizes, device=self.conv_gates.weight.device))]
    
    def forward(self, input_tensor, cur_state):
        """

        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        h_cur = cur_state[0]
        input_tensor = self.drop_out(input_tensor)
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return [h_next]

class GRUCell(BasicGRUCell):

    def __init__(self, input_dim: int, hidden_dim: int, drop_out: float = 0.2, bias: bool = True) -> None:
        super().__init__(input_dim, hidden_dim)

        self.conv_gates = nn.Linear(in_features=input_dim+hidden_dim, out_features=2*self.hidden_dim, bias=bias)
        self.conv_can   = nn.Linear(in_features=input_dim+hidden_dim, out_features=self.hidden_dim, bias=bias)

        if drop_out > 0.:
            self.drop_out = nn.Dropout(p=drop_out)

class BiGRUCell(GRUCell):

    def __init__(self, input_dim: int, hidden_dim: int, drop_out: float = 0.2, bias: bool = True) -> None:
        super().__init__(input_dim, hidden_dim, drop_out, bias)

        self.conv_concat = nn.Sequential(
            nn.Linear(in_features=2*hidden_dim, out_features=hidden_dim, bias=bias),
            nn.Tanh())

class ConvGRUCell(BasicGRUCell):

    def __init__(self, input_dim: int,
                       hidden_dim: int,
                       kernel_size: int = 3, 
                       stride: int = 1, 
                       padding: int = 1, 
                       bias: bool = True) -> None:
        super().__init__(input_dim, hidden_dim)

        self.conv_gates = nn.Conv1d(in_channels=input_dim + hidden_dim,
                                    out_channels=2 * self.hidden_dim,
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    bias=bias)

        self.conv_can  = nn.Conv1d(in_channels=input_dim + hidden_dim,
                              out_channels=self.hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)
        
class BiConvGRUCell(ConvGRUCell):

    '''
    
    
    '''
    def __init__(self, input_dim: int, hidden_dim: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 1, 
                 kernel_size_cat: int = 1, stride_cat: int = 1, padding_cat: int = 0,
                 bias=True):
        super().__init__(input_dim, hidden_dim, kernel_size, stride, padding, bias)

        self.conv_concat = nn.Sequential(
            nn.Conv1d(in_channels=2 * hidden_dim,
                                out_channels=hidden_dim,
                                kernel_size=kernel_size_cat,
                                stride=stride_cat,
                                padding=padding_cat,
                                bias=bias),
            nn.Tanh())

class LSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dims, cell_type: str, bi_direction: bool = True, kernel_sizes: List = None,
                 image_size = [], batch_first=False, bias=True, return_all_layers=False):
        super().__init__()

        # _check_kernel_size_consistency(kernel_sizes)
        self.num_layers = len(hidden_dims)
        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        # kernel_size = _extend_for_multilayer(kernel_sizes, self.num_layers)
        # hidden_dim = _extend_for_multilayer(hidden_dims, self.num_layers)
        # if not len(kernel_sizes) == len(hidden_dims) == self.num_layers:
        #     raise ValueError('Inconsistent list length.')

        # self.input_dim = input_dim
        # self.hidden_dim = hidden_dim
        # self.kernel_size = kernel_size
        # self.num_layers = num_layers
        # self.batch_first = batch_first
        # self.bias = bias
        self.return_all_layers = return_all_layers
        if bi_direction:
            # if cell_type in ['LSTM']:       cell_class = BiLSTMCell
            if cell_type in ['GRU']:        cell_class = BiGRUCell
            if cell_type in ['ConvLSTM']:   cell_class = BiConvLSTMCell
            if cell_type in ['ConvGRU']:    cell_class = BiConvGRUCell
            self.iter_method = self.iter_inner_bi
        else:
            if cell_type in ['LSTM']:       cell_class = LSTMCell
            if cell_type in ['GRU']:        cell_class = GRUCell
            if cell_type in ['ConvLSTM']:   cell_class = ConvLSTMCell
            if cell_type in ['ConvGRU']:    cell_class = ConvGRUCell
            self.iter_method = self.iter_inner

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            if cell_type in ['ConvGRU', 'ConvLSTM']:
                cell_list.append(cell_class(input_dim=cur_input_dim,
                                            hidden_dim=hidden_dims[i],
                                            kernel_size=kernel_sizes[i],
                                            bias=bias))
            else:
                cell_list.append(cell_class(input_dim=cur_input_dim,
                                            hidden_dim=hidden_dims[i],
                                            bias=bias))                

        self.cell_list = nn.ModuleList(cell_list)
        self.image_size = image_size

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        # if not self.batch_first:
        #     # (t, b, c, h, w) -> (b, t, c, h, w)
        #     input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b = input_tensor.size(0)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b, image_size=self.image_size)

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            layer_output, out_hidden_state = self.iter_method(cur_layer_input, layer_idx, seq_len, hidden_state[layer_idx])
            layer_output_list.append(layer_output)
            last_state_list.append(out_hidden_state)
            cur_layer_input = layer_output

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1]
            last_state_list = last_state_list[-1]

        return layer_output_list, last_state_list

    def iter_inner(self, cur_layer_input, layer_idx, seq_len, hidden_state):

        output_inner = []
        for t in range(seq_len):
            hidden_state = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :],
                                                cur_state=hidden_state)
            output_inner.append(hidden_state[0])

        layer_output = torch.stack(output_inner, dim=1)
        return layer_output, hidden_state

    def iter_inner_bi(self, cur_layer_input, layer_idx, seq_len, hidden_state):
        
        output_inner = []
        backward_states = []
        forward_states = []

        hidden_state_back = hidden_state
        hidden_state_forward = hidden_state
        for t in range(seq_len):
            hidden_state_back = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, seq_len - t - 1],
                                                cur_state=hidden_state_back)
            backward_states.append(hidden_state_back[0])

        for t in range(seq_len):
            hidden_state_forward = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t],
                                                cur_state=hidden_state_forward)
            forward_states.append(hidden_state_forward[0])
        
        for t in range(seq_len):
            h = self.cell_list[layer_idx].conv_concat(torch.cat((forward_states[t], backward_states[seq_len - t - 1]), dim=1))
            output_inner.append(h)

        layer_output = torch.stack(output_inner, dim=1)
        return layer_output, [backward_states, forward_states]

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states
