import time
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        input_size = 13800
        output_size = 8
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
 
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x
