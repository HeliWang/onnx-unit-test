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
import torch.autograd as autograd
import torch.nn.functional as F

sym, arg_params, aux_params = onnx_mxnet.import_model('test.onnx')

word_to_ix = {"hello": 0, "world": 1}

model = torch.load('test.pt')
np_input = [[word_to_ix["world"]]]
lookup_tensor = torch.LongTensor(np_input)
a = torch.autograd.Variable(lookup_tensor)

print("input:")
print(a.data.numpy())

print("input shape:")
print(a.data.numpy().shape)

print("PyTorch output (batch, sent_len, embed_dim):")
print(model(a).data.numpy())

print("MxNet - The input Symbol:")	
print(sym)

print("MxNet - Model parameter, dict of name to NDArray of net's weights:")
print(arg_params)

param_0 = mx.nd.array(np_input)
input_0 = mx.nd.array([[0.66135216, 0.26692411, 0.06167726, 0.62131733, -0.45190597], [-0.16613023, -1.5227685, 0.38168392, -1.02760863, -0.56305277]])
	
print("MxNet - Model parameter, dict of name to NDArray of net's auxiliary states")
print(aux_params)

mod = mx.mod.Module(symbol=sym, data_names=['input_0', 'param_0'], context=mx.cpu(), label_names=None)
mod.bind(for_training=False, data_shapes=[('input_0', tuple([1, 2, 5])), ('param_0', tuple([1, 1]))], label_shapes=None)
mod.set_params(arg_params=arg_params, aux_params=aux_params)

print("ONNX_MXNET Output:")
nd_iter = mx.io.NDArrayIter(data={'input_0': input_0, 'param_0': param_0}, label=None, batch_size=1)

for batch in nd_iter:
  mod.forward(batch)

print(mod.get_outputs())
