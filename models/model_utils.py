import torch

def initialize_weights(model, method='xavier'):
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
            if method == 'xavier':
                torch.nn.init.xavier_uniform_(layer.weight)
            elif method == 'kaiming':
                torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
