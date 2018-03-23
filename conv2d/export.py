import sys
import os
import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.onnx
from torch import nn
from model import Model
import torch.nn.functional as F


model = Model()
torch.save(model, 'test.pt')
a = torch.autograd.Variable(torch.FloatTensor(*(1, 2, 5, 10)))
export_file = "test.onnx"
torch.onnx.export(model, a, export_file, verbose=True)
