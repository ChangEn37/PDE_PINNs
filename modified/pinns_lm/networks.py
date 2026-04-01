import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, layers, last_bias=False):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(len(layers) - 1):
            is_last = i == len(layers) - 2
            use_bias = last_bias if is_last else True
            self.layers.append(nn.Linear(layers[i], layers[i + 1], bias=use_bias))

        self.activation = torch.tanh
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, inputs):
        value = inputs
        for layer in self.layers[:-1]:
            value = self.activation(layer(value))
        return self.layers[-1](value)

