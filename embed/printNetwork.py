import mxnet as mx
import numpy as np
import logging
import pprint
import mxnet.contrib.onnx as onnx_mxnet
from model import Model
from collections import namedtuple
import sys
import os
import random
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

sym, arg_params, aux_params = onnx_mxnet.import_model('test.onnx')
word_to_ix = {"hello": 0, "world": 1}
lookup_tensor = np.array([word_to_ix["hello"]])
print(sym)
mx.viz.plot_network(sym, shape={"input_0": lookup_tensor.shape}, node_attrs={"shape":'oval',"fixedsize":'false'})
