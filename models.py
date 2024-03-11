import torch.nn as nn
from torch.nn.utils.parametrizations import orthogonal
from torchsummary import summary
from torchview import draw_graph
from torch import device, cuda, reshape
import matplotlib.pyplot as plt
import numpy as np
import os

# TODO: at some point, create submodules for ff_models
# and submodules for encoders and decoders?

class SingleLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, orthogonality=False):
        super(SingleLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if orthogonality:
            self.linear = orthogonal(nn.Linear(in_dim, out_dim, bias=bias))
        else:
            self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.to(self.device)
    # end init

    def forward(self, inp):
        a = inp.view(-1, self.in_dim)
        return self.linear( a )
    # end forward

    def summary(self):
        summary(self, (1,self.in_dim))
    # end summary

    def plot_model(self):
        self.model_graph = draw_graph(self, input_size=(1,self.in_dim), \
            expand_nested=True, graph_name='single_layer', save_graph=True, directory='./figs')
        self.model_graph.visual_graph
    # end plot_model

    def visualize_weights(self):
        os.makedirs('figs', exist_ok=True)
        dir_name = 'figs/single_layer_weights'
        os.makedirs(dir_name, exist_ok=True)
        for i in range(self.out_dim):
            plt.clf()
            plt.plot( self.linear.weight[i,:].cpu().detach().numpy() )
            plt.savefig(dir_name + '/weights_' + str(i) + '.png', dpi=150)
    # end visualize_weights
# end class SingleLayer