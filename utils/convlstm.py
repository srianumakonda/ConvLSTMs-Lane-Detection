import torch
import torch.nn as nn

class ConvLSTM_Cell(nn.Module):

    def __init__(self,input_dim,hidden_dim,kernel_size,bias):

        super(ConvLSTM_Cell,self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size//2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim+self.hidden_dim,
            out_channels=4*self.hidden_dim, #multiplied by 4 because there are 4 gates in an LSTM cell
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):

        if cur_state is None:
            cur_state = self._init_hidden(self.batch_size,self.img_dim)

        hidden, current = cur_state

        conc = torch.cat([input_tensor, hidden], dim=1)
        conv = self.conv(conc)

        i, f, o, g = torch.split(conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        current_next = f * current + i * g
        hidden_next = o * torch.tanh(current_next)

        return hidden_next, current_next

    def _init_hidden(self, batch_size, input_shape): 
        
        height, width = input_shape
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):

    def __init__(self,input_dim,hidden_dim,kernel_size,num_layers,bias,return_layers=False):

        super(ConvLSTM,self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias
        self.return_layers = return_layers

        cell_list = []
        for i in range(0, self.num_layers):
            if i==0:
                current_input_dim = self.input_dim
            else:
                current_input_dim = self.hidden_dim[i-1]
            
            cell_list.append(ConvLSTM_Cell(input_dim=current_input_dim,
                                           hidden_dim=self.hidden_dim[i],
                                           kernel_size=self.kernel_size,
                                           bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):

        batch, _, _, height, width = input_tensor.size()

        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=batch,
                                             input_shape=(height,width))

        output_layer = []
        final_state = []

        seq_len = input_tensor.size(1)
        current_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            hidden, current = hidden_state[layer_idx]
            inner_output = []
            for t in range(seq_len):
                hidden, current = self.cell_list[layer_idx](input_tensor=current_layer_input[:,t,:,:,:], 
                                                            cur_state=[hidden,current])
                inner_output.append(hidden)
            
            layer_output = torch.stack(inner_output,dim=1)
            current_layer_input = layer_output

            output_layer.append(layer_output)
            final_state.append([hidden,current])

        if not self.return_layers:
            output_layer = output_layer[-1:]
            final_state = final_state[-1:]

        return output_layer, final_state

    def _init_hidden(self, batch_size, input_shape):
        
        init_state = []
        for i in range(self.num_layers):
            init_state.append(self.cell_list[i]._init_hidden(batch_size, input_shape))
        
        return init_state

