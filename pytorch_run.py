import sys
import os
import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.onnx
from torch import nn

class Model(nn.Module):
   def __init__(self):
       super(Model, self).__init__()
       self.cnn = nn.Conv2d(
        in_channels=2,
        out_channels=5,
        kernel_size=(3, 10),
        padding=(2, 0)
       )
   def forward(self, x):
       y = self.cnn(x)
       return y


model = Model()
a = torch.autograd.Variable(torch.FloatTensor(*(1, 2, 5, 10)))
print(a.data.numpy())
print(model(a))
