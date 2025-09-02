import torch
from config import get_args

args = get_args()



class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, negative_slope=0.01):
        super(Feedforward, self).__init__()
        
        layers = []
        in_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(torch.nn.Linear(in_size, hidden_size))
            layers.append(torch.nn.LeakyReLU(negative_slope=negative_slope))
            in_size = hidden_size
        
        layers.append(torch.nn.Linear(in_size, num_classes))
        
        if num_classes == 1:
            layers.append(torch.nn.Sigmoid())
        
        self.network = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


