import numpy as np
import mxnet as mx
import mxnet.contrib.onnx as onnx_mxnet
import torch.nn.functional as F

sym, arg_params, aux_params = onnx_mxnet.import_model('cnn_test.onnx')

random_input = np.random.rand(1, 2, 5, 10)
mod = mx.mod.Module(symbol=sym, data_names=['input_0'], context=mx.cpu(), label_names=None)
#print(random_input.shape)
mod.bind(for_training=False, data_shapes=[('input_0', random_input.shape)], label_shapes=None)
mod.set_params(arg_params=arg_params, aux_params=aux_params)

# Forward method needs Batch of data as input
from collections import namedtuple
# run inference
# Batch = namedtuple('Batch', ['data'])
# mod.forward(Batch([mx.nd.array(random_input)]))

# print(mod.get_outputs())
#print(mod.get_outputs()[0].asnumpy())

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
       z = F.max_pool2d(y, kernel_size=(y.size(2), y.size(3)))
       return z


model = torch.load('cnn_test.pt')
a = torch.autograd.Variable(torch.FloatTensor(*(1, 2, 5, 10)))
print("input:")
print(a.data.numpy())

print("PyTorch Output:")
print(model(a).data.numpy())

print("ONNX_MXNET Output:")
Batch = namedtuple('Batch', ['data'])
mod.forward(Batch([mx.nd.array(a.data.numpy())]))

print(mod.get_outputs())
#print(mod.get_outputs()[0].asnumpy())
