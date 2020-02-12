import time
import torch.nn as nn
import torch.nn.functional as F

class NewModel(nn.Module):
    def __init__(self):
        super(NewModel, self).__init__()
        self.conv = nn.Conv2d(1,40*32,3)
        self.batch = nn.BatchNorm2d(40*32)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(4158720, 8)
        self.drop = nn.Dropout2d(0.2)

    def forward(self, x):
        print('--->')
        
        x = x.transpose(3,1)
        x = self.drop(self.pool(F.relu(self.batch(self.conv(x)))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
        #return F.log_softmax(x, dim = 0)