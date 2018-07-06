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
import evaluate

print >> sys.stderr, "PID", os.getpid()
random.seed(0)
numpy.random.seed(0)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
torch.cuda.set_device(args.gpu)

MAX = 2

def main():
    fix=""
    if args.reduced == 1:
        fix="_reduced"
    read_f = file("./data/train_data"+fix,"rb")
    train_generater = cPickle.load(read_f)
    read_f.close()
    read_f = file("./data/emb","rb")
    embedding_matrix,_,_ = cPickle.load(read_f)
    read_f.close()
    test_generater = DataGnerater("test",256)

    print "Building torch model"
    model = Network(nnargs["embedding_size"],nnargs["embedding_dimention"],embedding_matrix,nnargs["hidden_dimention"]).cuda()

    this_lr = 0.00003
    optimizer = optim.Adam(model.parameters(), lr=this_lr)
    for echo in range(nnargs["epoch"]):
        cost = 0.0
        print >> sys.stderr, "Begin epoch",echo
        for doc in train_generater.generate_data(shuffle=True):
            loss = 0.0
            for gold,score,zp_indexs,np_indexs in model.forward(doc,dropout=0.0):
                loss += (-1.0*torch.log(torch.sum(gold*score)+1e-12)\
                        + -1.0*torch.log(1.0-torch.sum((1-gold)*score)+1e-12))
            optimizer.zero_grad()
            cost += loss.item()
            loss.backward()
            optimizer.step()
        print >> sys.stderr, "End epoch",echo,"Cost:", cost
        with torch.no_grad():
            #result = evaluate.evaluate(model,train_generater,dev=True)
            result = evaluate.evaluate(model,train_generater,dev=False)
            print result["all_pos"],result["predict_pos"],result["hit"]
if __name__ == "__main__":
    main()
