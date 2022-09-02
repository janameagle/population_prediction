import torch.nn as nn
import torch
import numpy as np

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor. # number of driving factors?
        hidden_dim: int
            Number of channels of hidden state. # ?
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        #self.padding = kernel_size[0] // 2, kernel_size[1] // 2 # paddings adds a frame, so that image size stays the same after moving window
        self.padding = kernel_size // 2, kernel_size// 2 # paddings adds a frame, so that image size stays the same after moving window
        
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels = 4 * self.hidden_dim, # why 4? because 4 are needed for LSTM? i, f, o, g? Or 4 time steps?
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) # Splits the tensor into chunks. Each chunk is a view of the original tensor.
        # input, forget, output, and cell gates (corresponding to torch's LSTM)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    
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
    
    
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

 # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.time_steps = 4

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)


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

        if not self.batch_first: # change order if batch size is not first dimension
            # (t, b, c, h, w) -> (b, t, c, h, w) # batch size, time, channel, height, width
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=b,image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1) # t, number of years
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers): # loop over layers

            h, c = hidden_state[layer_idx] # images with several channels
            output_inner = []
            for t in range(seq_len): # loop over time steps
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])   # cell_list ist a list of 1 convlstm for every layer
                output_inner.append(h)


            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param



###############################################################################
# Convolutional bidirectional LSTM

# class ConvBLSTM(nn.Module):
#     def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, 
#                  batch_first=False, bias=True, return_all_layers=False):
#         super(ConvBLSTM, self).__init__()
    
#         self.return_all_layers = return_all_layers
        
#         self.forward_cell = ConvLSTM(input_dim, hidden_dim, 
#                                    kernel_size, num_layers, 
#                                    batch_first=False, bias=True, return_all_layers=False)
#         self.backward_cell = ConvLSTM(input_dim, hidden_dim, 
#                                    kernel_size, num_layers, 
#                                    batch_first=False, bias=True, return_all_layers=False)  


#     def forward(self, x):
#         y_out_forward = self.forward_cell(x)[0]
#         x_reversed = torch.flip(x, [1])
#         y_out_reverse = self.backward_cell(x_reversed)[0]
#         output = torch.cat((y_out_forward, y_out_reverse), dim=2)
#         if not self.return_all_layers:
#             output = torch.squeeze(output[:, -1,...], dim=1)
#         return output
        
   
class ConvBLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, #num_layers, #img_size?
                  batch_first=False, bias=True, return_all_layers=False):
        super(ConvBLSTM, self).__init__()
    
        self.return_all_layers = return_all_layers
        self.batch_first = batch_first
        
        self.cell_fw = ConvLSTMCell(input_dim=input_dim,
                                      hidden_dim=hidden_dim,
                                      kernel_size=kernel_size,
                                      bias=bias)
        
        self.cell_bw = ConvLSTMCell(input_dim=input_dim,
                                      hidden_dim=hidden_dim,
                                      kernel_size=kernel_size,
                                      bias=bias) 
        
        self.lstm = ConvLSTMCell(input_dim=hidden_dim*2,
                    hidden_dim=1, # to create a regression output
                    kernel_size=kernel_size,
                    bias=bias)
        

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first: # change order if batch size is not first dimension
            # (t, b, c, h, w) -> (b, t, c, h, w) # batch size, time, channel, height, width
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b, seq_len, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state, hidden_state_inv = self._init_hidden(batch_size=b, image_size = (h,w))
        
        ## LSTM forward direction
        h, c = hidden_state
        output_inner = []
        for t in range(seq_len):
            h, c = self.cell_fw(input_tensor=input_tensor[:, t, :, :, :],
                                             cur_state=[h, c])
            
            output_inner.append(h)
        output_inner = torch.stack((output_inner), dim=1)
        layer_output = output_inner
        last_state = [h, c]
        ####################

        ## LSTM inverse direction
        input_inv = input_tensor
        h_inv, c_inv = hidden_state_inv
        output_inv = []
        for t in range(seq_len-1, -1, -1):
            h_inv, c_inv = self.cell_bw(input_tensor=input_inv[:, t, :, :, :],
                                             cur_state=[h_inv, c_inv])
            
            output_inv.append(h_inv)
        output_inv.reverse() 
        output_inv = torch.stack((output_inv), dim=1)
        layer_output = torch.cat((output_inner, output_inv), dim=2)
        last_state_inv = [h_inv, c_inv]
    ###################################
      
        
        ## standard LSTM layer
        h, c = self._init_hidden_lstm(batch_size=b,image_size=(input_tensor.size(3), input_tensor.size(4)))
        
        output_lstm = []
        for t in range(seq_len): # loop over time steps
            h, c = self.lstm(input_tensor = layer_output[:,t,:,:,:],#?
                             cur_state = [h, c])    
            output_lstm.append(h)
    
        total_output = torch.stack(output_lstm, dim=1)
    
        # if not self.return_all_layers:
        #     layer_output_list = layer_output_list[-1:]
        #     last_state_list = last_state_list[-1:]
        
        return total_output, [h,c]
    
        # return layer_output if self.return_all_layers is True else layer_output[:, -1:], last_state, last_state_inv


    def _init_hidden(self, batch_size, image_size):
            init_states_fw = self.cell_fw.init_hidden(batch_size, image_size)
            init_states_bw = None
            init_states_bw = self.cell_bw.init_hidden(batch_size, image_size)
            return init_states_fw, init_states_bw

    def _init_hidden_lstm(self, batch_size, image_size):
        init_states = self.lstm.init_hidden(batch_size, image_size)
        return init_states
   
# x1 = torch.randn([4, 10, 256, 256]) #(t, c, w, h)
# x2 = torch.randn([4, 10, 256, 256])

# cblstm = ConvBLSTM(input_dim=10, hidden_dim=[32, 1], kernel_size=(3, 3), num_layers = 2)

# x = torch.stack([x1, x2], dim=1)
# print(x.shape)
# out = cblstm(x)
# print (out.shape)
# out.sum().backward()    


