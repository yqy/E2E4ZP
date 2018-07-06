#coding=utf8
import os
import sys
import re
import argparse
import math
import timeit
import numpy
import random
import cPickle
sys.setrecursionlimit(1000000)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision.transforms as T
from torch.optim import lr_scheduler

from conf import *
import utils
from data_generater import *
from net import *

random.seed(0)
numpy.random.seed(0)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
torch.cuda.set_device(args.gpu)

MAX = 2

def evaluate(model,data_generater,dev=False):
    all = 0
    all_in_all = 0
    predict_all = 0
    for doc in data_generater.generate_data(dev=dev):
        all += len(doc.all_azps)
        for gold,score,zp_indexs,np_indexs in model.forward(doc,dropout=0.0):
            this_score = score.data.cpu().numpy()
            max_index = numpy.argmax(this_score)
            if max_index < (len(this_score)-1):
                predict_all += 1
                this_np_index = np_indexs[max_index]
                if doc.zp_candi_coref[zp_indexs][this_np_index] == 1:
                    all_in_all += 1 
    result = {}
    result["all_pos"] = all
    result["predict_pos"] = predict_all
    result["hit"] = all_in_all
    r = float(all_in_all)/float(all) if all > 0 else 0.0
    p = float(all_in_all)/float(predict_all) if predict_all > 0 else 0.0
    f = 2.0/(1.0/r+1.0/p) if (r > 0 and p > 0) else 0.0
    result["r"] = r
    result["p"] = p
    result["f"] = f
    return result

