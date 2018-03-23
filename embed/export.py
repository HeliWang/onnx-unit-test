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
import torch.optim as optim

torch.manual_seed(1)
word_to_ix = {"hello": 0, "world": 1}
lookup_tensor = torch.LongTensor([[word_to_ix["hello"]]]) # output will be (batch, sent_len, embed_dim)

model = Model()
a = torch.autograd.Variable(lookup_tensor)

hello_embed = model(a)
print(hello_embed)

export_file = "test.onnx"
torch.save(model, 'test.pt')
torch.onnx.export(model, a, export_file, verbose=True)
