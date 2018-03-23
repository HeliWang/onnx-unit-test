import sys
import os
import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.onnx
from torch import nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

torch.manual_seed(1)
words_num = 2
words_dim = 5

class Model(nn.Module):
   def __init__(self):
       super(Model, self).__init__()
       self.embed = nn.Embedding(words_num, words_dim)
   def forward(self, x):
       word_input = self.embed(x) # output is (batch, sent_len, embed_dim)
       return word_input
