import numpy as np
import mxnet as mx
import mxnet.contrib.onnx as onnx_mxnet
import torch.nn.functional as F
from torch import nn
from model import Model
from collections import namedtuple
import sys
import os
import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from numpy import array


sym, arg_params, aux_params = onnx_mxnet.import_model('test.onnx')

random_input = np.random.rand(1, 2, 5, 10)
mod = mx.mod.Module(symbol=sym, data_names=['input_0'], context=mx.cpu(), label_names=None)
#print(random_input.shape)
mod.bind(for_training=False, data_shapes=[('input_0', random_input.shape)], label_shapes=None)
mod.set_params(arg_params=arg_params, aux_params=aux_params)

model = torch.load('test.pt')
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
