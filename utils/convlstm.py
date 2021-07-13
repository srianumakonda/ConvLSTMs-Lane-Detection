import torch
import torch.nn as nn

class ConvLSTM_Cell(nn.Module):

    def __init__(self,input_shape,input_dim,hidden_dim,kernel_size,bias):

        super(ConvLSTM_Cell,self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.height, self.width = input_shape
        self.kernel_size = kernel_size
        self.bias = bias
        self.padding = kernel_size[0]//2

        self.conv = nn.Conv2d(
            in_channels=self.input_dim+self.hidden_dim,
            out_channels=4*self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_state, state):

        if state is None:
            state = self._init_hidden(self.batch_size,self.img_dim)

        hidden, current = state

        conc = torch.cat([input_state,hidden],dim=1)
        conv = self.conv(conc)

        i, f, o, g = torch.split(conv,self.hidden_dim,dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        current_next = f * current + i * g
        hidden_next = o * torch.tanh(current_next)

        return hidden_next, current_next

    def _init_hidden(self, batch_size): #change back to init_hidden in case that it doesnt work (get rid of underscore)
        return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):

    def __init__(self,input_shape,input_dim,hidden_dim,kernel_size,bias):

        super(ConvLSTM,self).__init__()

        self.height, self.width = input_shape
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias

        self.convlstm = ConvLSTM_Cell(
            input_shape=(self.height,self.width),
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            bias=self.bias)

    def forward(self, input_state, state):

        hidden_state = self.

