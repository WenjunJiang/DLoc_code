import torch
import torch.nn as nn
import torch.nn.init
import functools
import torch.optim.lr_scheduler
# from util.image_pool import ImagePool
# from collections import OrderedDict
# import time
# from options.train_options import TrainOptions
# from collections import defaultdict
# import h5py
# import scipy.io
# from torch.autograd import Variable
# import torch.optim as optim
import numpy as np


# import torchvision
import os
# from easydict import EasyDict as edict
# import random
# import matplotlib.pyplot as plt
# import sys
# import ntpath
# import time
# from scipy.misc import imresize
# import json

import Generators
# from LocationNetworks import *

def write_log(log_values, model_name, log_dir="", log_type='loss', type_write='a'):
    if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    with open(log_dir + os.path.sep + model_name+ "_" +log_type + ".txt", type_write) as f:
        f.write(','.join(log_values)+"\n")

def get_model_funct(model_name):
    if model_name == "G":
        return define_G
    else:
        # TODO
        raise NotImplementedError("Model Type %s does not exist."%model_name)

# def define_L(opt, gpu_ids):
#     return LocationNetwork()

def define_G(opt, gpu_ids):
    net = None
    norm_layer = get_norm_layer(norm_type=opt.norm)

    # if opt.base_model == 'resnet_9blocks':
    #     net = Generators.ResnetGenerator(opt.input_nc, opt.output_nc, opt.ngf, \
    #                                      norm_layer=norm_layer, \
    #                                      use_dropout=opt.no_dropout, \
    #                                      n_blocks=9)
    if opt.base_model == 'resnet_nblocks':
        # n_blocks    = opt.resnet_blocks
        net = Generators.ResnetGenerator(opt.input_nc, opt.output_nc, opt.ngf, \
                                         norm_layer=norm_layer, \
                                         use_dropout=opt.no_dropout, \
                                         n_blocks= opt.resnet_blocks)
    elif opt.base_model == 'resnet2_nblocks':
        # n_blocks    = opt.resnet_blocks
        net = Generators.ResnetGenerator2(opt.input_nc, opt.output_nc, opt.ngf, \
                                          norm_layer=norm_layer, \
                                          use_dropout=opt.no_dropout, \
                                          n_blocks= opt.resnet_blocks)
    # elif opt.base_model == 'resnet_6blocks':
    #     net = Generators.ResnetGenerator(opt.input_nc, opt.output_nc, opt.ngf, \
    #                                      norm_layer=opt.norm_layer, \
    #                                      use_dropout=opt.no_dropout, \
    #                                      n_blocks=6)
    elif opt.base_model == 'resnet_encoder':
        # n_blocks    = opt.resnet_blocks
        net = Generators.ResnetEncoder(opt.input_nc, opt.output_nc, opt.ngf, \
                                       norm_layer=norm_layer, \
                                       use_dropout=opt.no_dropout, \
                                       n_blocks= opt.resnet_blocks)
    elif opt.base_model == 'resnet_decoder':
        # n_blocks    = opt.resnet_blocks
        net = Generators.ResnetDecoder(opt.input_nc, opt.output_nc, opt.ngf, \
                                       norm_layer=norm_layer, \
                                       use_dropout=opt.no_dropout, \
                                       n_blocks= opt.resnet_blocks, \
                                       encoder_blocks=opt.encoder_res_blocks)
    elif opt.base_model == 'unet_128':
        net = Generators.UnetGenerator(opt.input_nc, opt.output_nc, 7, opt.ngf, \
                                       norm_layer=norm_layer, \
                                       use_dropout=opt.no_dropout)
    elif opt.base_model == 'unet_256':
        net = Generators.UnetGenerator(opt.input_nc, opt.output_nc, 8, opt.ngf, \
                                       norm_layer=norm_layer, \
                                       use_dropout=opt.no_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' \
                                  %(opt.base_model))
    return init_net(net, opt.init_type, opt.init_gain, gpu_ids)

def get_scheduler(optimizer, opt):
    if opt.starting_epoch_count=='best' and opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            print("lambda update %s, %s, %s", (epoch, 0))
            lr_l = 1.0 - max(0, epoch + 1 - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.starting_epoch_count!='best' and opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            print("lambda update %s, %s, %s", (epoch, opt.starting_epoch_count))
            lr_l = 1.0 - max(0, epoch + 1 + opt.starting_epoch_count - opt.niter) \
                            / float(opt.niter_decay + 1)
            return lr_l
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.starting_epoch_count!='best' and opt.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, \
                                                    step_size=opt.lr_decay_iters, \
                                                    gamma=0.9)
    elif opt.starting_epoch_count!='best' and opt.lr_policy == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', \
                                                               factor=0.2, \
                                                               threshold=0.01, \
                                                               patience=5)
    elif opt.starting_epoch_count!='best' and opt.lr_policy == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, \
                                                               T_max=opt.niter, \
                                                               eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented' \
                                   %(opt.lr_policy))
    return scheduler

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, \
                                       track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def init_weights(net, init_type='normal', gain=1):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0, gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, gain)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=1, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        device_ = torch.device('cuda:{}'.format(gpu_ids[0]))
#         net.to(d)
        gpu_ids_int = list(map(int,gpu_ids))
        net = torch.nn.DataParallel(net, gpu_ids_int)
        net.to(device_)
    init_weights(net, init_type, gain=init_gain)
    return net

def localization_error(output_predictions,input_labels,scale=0.1):
    outputs = np.squeeze(output_predictions)
    inputs = np.squeeze(input_labels)
    image_size = outputs.shape
    error = np.zeros((image_size[0]))
    if(image_size[0]==161):
        label_temp = inputs
        pred_temp = outputs
        label_index = np.asarray(np.unravel_index(np.argmax(label_temp), label_temp.shape))
        prediction_index = np.asarray(np.unravel_index(np.argmax(pred_temp),pred_temp.shape))
        error[0] = np.sqrt( np.sum( np.power(np.multiply( label_index-prediction_index, scale ), 2)) )
        return error
    else:
        for i in range(image_size[0]):
            label_temp = inputs[i,:,:].squeeze()
            pred_temp = outputs[i,:,:].squeeze()
            label_index = np.asarray(np.unravel_index(np.argmax(label_temp), label_temp.shape))
            prediction_index = np.asarray(np.unravel_index(np.argmax(pred_temp),pred_temp.shape))
            error[i] = np.sqrt( np.sum( np.power(np.multiply( label_index-prediction_index, scale ), 2)) )
        return error

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
