#!/usr/bin/python

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
# from util.image_pool import ImagePool
from collections import OrderedDict
import time
# from options.train_options import TrainOptions
from collections import defaultdict
import h5py
import scipy.io
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import torchvision
import os
from easydict import EasyDict as edict
import random
import matplotlib.pyplot as plt
import sys
import ntpath
import time
from scipy.misc import imresize
import json

from utils import *
from modelADT import ModelADT
from Generators import *
from LocationNetworks import *
from data_loader import *
from params import *


class Enc_Dec_Network():

    def initialize(self, encoder, decoder, frozen_dec=False, frozen_enc=False, gpu_ids='1'):
        self.encoder = encoder
        self.decoder = decoder
        self.frozen_dec = frozen_dec
        self.frozen_enc = frozen_enc
        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) # if self.gpu_ids else torch.device('cpu')
        # self.encoder.net = encoder.net.to(self.device)
        # self.decoder.net = decoder.net.to(self.device)

    def set_input(self, input, target, convert_enc=True, shuffle_channel=True):
        self.input = input.to(self.device)
        self.target = target.to(self.device)
        self.encoder.set_data(self.input, self.input, convert=convert_enc, shuffle_channel=shuffle_channel)
    
    def save_networks(self, epoch):
        self.encoder.save_networks(epoch)
        self.decoder.save_networks(epoch)

    def save_outputs(self):
        self.encoder.save_outputs()
        self.decoder.save_outputs()
    
    def update_learning_rate(self):
        self.encoder.update_learning_rate()
        
    def forward(self):
        self.encoder.forward()
        self.decoder.set_data(self.encoder.output, self.target)
        self.decoder.forward()
    
    def test(self):
        self.encoder.test()
        self.decoder.set_data(self.encoder.output, self.target)
        self.decoder.test()

    def backward(self):
        self.decoder.backward()
        # self.encoder.backward()
        
    def optimize_parameters(self):
        self.forward()
        self.backward()
        if not self.frozen_enc:
            self.encoder.optimizer.step()
        if not self.frozen_dec:
            self.decoder.optimizer.step()
        self.encoder.optimizer.zero_grad()
        self.decoder.optimizer.zero_grad()

    def eval(self):
        self.encoder.forward()
        self.decoder.set_data(self.encoder.output, self.target)
        self.decoder.forward()



class Enc_2Dec_Network():

    def initialize(self, opt , encoder, decoder, offset_decoder, frozen_dec=False, frozen_enc=False, gpu_ids='1'):
        print('initializing Encoder and 2 Decoders Model')
        self.opt = opt
        self.isTrain = opt.isTrain
        self.encoder = encoder
        self.decoder = decoder
        self.offset_decoder = offset_decoder
        self.frozen_dec = frozen_dec
        self.frozen_enc = frozen_enc
        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) # if self.gpu_ids else torch.device('cpu')
        # self.encoder.net = encoder.net.to(self.device)
        # self.decoder.net = decoder.net.to(self.device)

    def set_input(self, input, target ,offset_target ,convert_enc=True, shuffle_channel=True):
        self.input = input.to(self.device)
        self.target = target.to(self.device)
        self.offset_target = offset_target.to(self.device)
        self.encoder.set_data(self.input, self.input, convert=convert_enc, shuffle_channel=shuffle_channel)
    
    def save_networks(self, epoch):
        self.encoder.save_networks(epoch)
        self.decoder.save_networks(epoch)
        self.offset_decoder.save_networks(epoch)

    def save_outputs(self):
        self.encoder.save_outputs()
        self.decoder.save_outputs()
        self.offset_decoder.save_outputs()
    
    def update_learning_rate(self):
        self.encoder.update_learning_rate()
        self.decoder.update_learning_rate()
        self.offset_decoder.update_learning_rate()
        
    def forward(self):
        self.encoder.forward()
        self.decoder.set_data(self.encoder.output, self.target)
        self.offset_decoder.set_data(self.encoder.output, self.offset_target)
        self.decoder.forward()
        self.offset_decoder.forward()
    
    # Test the network once set into Evaluation mode!
    def test(self):
        self.encoder.test()
        self.decoder.set_data(self.encoder.output, self.target)
        self.offset_decoder.set_data(self.encoder.output, self.offset_target)
        self.decoder.test()
        self.offset_decoder.test()

    def backward(self):
        self.decoder.backward()
        self.offset_decoder.backward()
        # self.encoder.backward()
        
    def optimize_parameters(self):
        self.forward()
        self.backward()
        if not self.frozen_enc:
            self.encoder.optimizer.step()
        if not self.frozen_dec:
            self.decoder.optimizer.step()
            self.offset_decoder.optimizer.step()
        self.encoder.optimizer.zero_grad()
        self.decoder.optimizer.zero_grad()
        self.offset_decoder.optimizer.zero_grad()

    # set the models to evaluation mode
    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        self.offset_decoder.eval()