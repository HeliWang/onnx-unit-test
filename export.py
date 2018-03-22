import sys
import os
import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.onnx
from torch import nn

import torch.nn.functional as F

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
       z = F.max_pool2d(y, kernel_size=(y.size(2), y.size(3)))
       return z

model = Model()
torch.save(model, 'cnn_test.pt')
a = torch.autograd.Variable(torch.FloatTensor(*(1, 2, 5, 10)))
export_file = "cnn_test.onnx"
torch.onnx.export(model, a, export_file, verbose=True)
