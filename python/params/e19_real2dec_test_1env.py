
import os
from easydict import EasyDict as edict
import json



opt_exp = edict()

opt_exp.isTrain = False
opt_exp.continue_train = False #action='store_true', help='continue training: load the latest model')
opt_exp.starting_epoch_count = 'best' #type=int, default=1, help='the starting epoch count, we save the model by <starting_epoch_count>, <starting_epoch_count>+<save_latest_freq>, ...')
opt_exp.isTrainGen = False
opt_exp.isTrainLoc = False
opt_exp.isFrozen = False
opt_exp.isFrozen_gen = False
opt_exp.n_epochs = 50
opt_exp.data = "1st_environment"
opt_exp.gpu_ids = ['1','2','3','0']

opt_exp.phase = 'rw_test_1env' #type=str, default='train', help='train, val, test, etc')
opt_exp.name = 'e19_real2dec_train' #type=str, default='train', help='train, val, test, etc')

opt_exp.checkpoints_dir = './checkpoints' + "/" + opt_exp.name#type=str, default='./checkpoints', help='models are saved here')
opt_exp.results_dir = './results' + "/" + opt_exp.name
opt_exp.log_dir = "./logs" + "/" + opt_exp.name
opt_exp.batch_size = 32


# In[13]:
# define base_options
opt_encoder = edict()

# hyperparams
# dataroot #'path to images (should have subfolders trainA, trainB, valA, valB, etc)')
opt_encoder.parent_exp = opt_exp
opt_encoder.batch_size = opt_encoder.parent_exp.batch_size #type=int, default=1, help='input batch size')
opt_encoder.ngf = 64 #type=int, default=64, help='# of gen filters in first conv layer')
opt_encoder.base_model = 'resnet_encoder' #type=str, default='resnet_9blocks', help='selects model to use for netG')
opt_encoder.net = 'G' #type=str, default='resnet_9blocks', help='selects model to use for netG')
opt_encoder.resnet_blocks = 6
opt_encoder.no_dropout = False #action='store_true', help='no dropout for the generator')
opt_encoder.init_type = 'xavier' #type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
opt_encoder.init_gain = 0.02 #type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
opt_encoder.norm = 'instance' #type=str, default='instance', help='instance normalization or batch normalization')
opt_encoder.beta1 = 0.5 #type=float, default=0.5, help='momentum term of adam')
opt_encoder.lr = 0.00001 #type=float, default=0.0002, help='initial learning rate for adam')
opt_encoder.lr_policy = 'step' #type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
opt_encoder.lr_decay_iters = 50 #type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
opt_encoder.lambda_L = 1 # weightage given to the Generator
opt_encoder.lambda_cross = 1e-5
opt_encoder.lambda_reg = 5e-4
opt_encoder.weight_decay = 1e-5


opt_encoder.input_nc = 4 #type=int, default=3, help='# of input image channels')
opt_encoder.output_nc = 1 #type=int, default=3, help='# of output image channels')
opt_encoder.save_latest_freq = 5000 #type=int, default=5000, help='frequency of saving the latest results')
opt_encoder.save_epoch_freq = 1 #type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
opt_encoder.n_epochs = opt_encoder.parent_exp.n_epochs
opt_encoder.isTrain = True
opt_encoder.continue_train = False #action='store_true', help='continue training: load the latest model')
opt_encoder.starting_epoch_count = opt_encoder.parent_exp.starting_epoch_count #type=int, default=1, help='the starting epoch count, we save the model by <starting_epoch_count>, <starting_epoch_count>+<save_latest_freq>, ...')
opt_encoder.phase = opt_encoder.parent_exp.phase #type=str, default='train', help='train, val, test, etc')
opt_encoder.name = 'encoder' #type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
opt_encoder.loss_type = "NoLoss"
opt_encoder.niter = 20 #type=int, default=100, help='# of iter at starting learning rate')
opt_encoder.niter_decay = 100 #type=int, default=100, help='# of iter to linearly decay learning rate to zero')



opt_encoder.gpu_ids = opt_encoder.parent_exp.gpu_ids #type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
opt_encoder.num_threads = 4 #default=4, type=int, help='# threads for loading data')
opt_encoder.checkpoints_load_dir =  opt_encoder.parent_exp.checkpoints_dir #type=str, default='./checkpoints', help='models are saved here')
opt_encoder.checkpoints_save_dir =  opt_encoder.parent_exp.checkpoints_dir #type=str, default='./checkpoints', help='models are saved here')
opt_encoder.results_dir = opt_encoder.parent_exp.results_dir
opt_encoder.log_dir =  opt_encoder.parent_exp.log_dir #type=str, default='./checkpoints', help='models are saved here')
opt_encoder.max_dataset_size = float("inf") #type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
opt_encoder.verbose = False  #action='store_true', help='if specified, print more debugging information')
opt_encoder.suffix ='' #default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')



# In[13]:
# define base_options
opt_decoder = edict()

# hyperparams
# dataroot #'path to images (should have subfolders trainA, trainB, valA, valB, etc)')
opt_decoder.parent_exp = opt_exp
opt_decoder.batch_size = opt_decoder.parent_exp.batch_size #type=int, default=1, help='input batch size')
opt_decoder.ngf = 64 #type=int, default=64, help='# of gen filters in first conv layer')
opt_decoder.base_model = 'resnet_decoder' #type=str, default='resnet_9blocks', help='selects model to use for netG')
opt_decoder.net = 'G' #type=str, default='resnet_9blocks', help='selects model to use for netG')opt_decoder.no_dropout = False #action='store_true', help='no dropout for the generator')
opt_decoder.resnet_blocks = 9
opt_decoder.encoder_res_blocks = 6
opt_decoder.no_dropout = False
opt_decoder.init_type = 'xavier' #type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
opt_decoder.init_gain = 0.02 #type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
opt_decoder.norm = 'instance' #type=str, default='instance', help='instance normalization or batch normalization')
opt_decoder.beta1 = 0.5 #type=float, default=0.5, help='momentum term of adam')
opt_decoder.lr = 0.00001 #type=float, default=0.0002, help='initial learning rate for adam')
opt_decoder.lr_policy = 'step' #type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
opt_decoder.lr_decay_iters = 50 #type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
opt_decoder.lambda_L = 1 # weightage given to the Generator
opt_decoder.lambda_cross = 1e-5
opt_decoder.lambda_reg = 5e-4
opt_decoder.weight_decay = 1e-5


opt_decoder.input_nc = 4 #type=int, default=3, help='# of input image channels')
opt_decoder.output_nc = 1 #type=int, default=3, help='# of output image channels')
opt_decoder.save_latest_freq = 5000 #type=int, default=5000, help='frequency of saving the latest results')
opt_decoder.save_epoch_freq = 1 #type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
opt_decoder.n_epochs = opt_decoder.parent_exp.n_epochs
opt_decoder.isTrain = True
opt_decoder.continue_train = False #action='store_true', help='continue training: load the latest model')
opt_decoder.starting_epoch_count = opt_decoder.parent_exp.starting_epoch_count #type=int, default=1, help='the starting epoch count, we save the model by <starting_epoch_count>, <starting_epoch_count>+<save_latest_freq>, ...')
opt_decoder.phase = opt_decoder.parent_exp.phase #type=str, default='train', help='train, val, test, etc')
opt_decoder.name = 'decoder' #type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
opt_decoder.loss_type = "L2_sumL1"
opt_decoder.niter = 20 #type=int, default=100, help='# of iter at starting learning rate')
opt_decoder.niter_decay = 100 #type=int, default=100, help='# of iter to linearly decay learning rate to zero')



opt_decoder.gpu_ids = opt_decoder.parent_exp.gpu_ids #type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
opt_decoder.num_threads = 4 #default=4, type=int, help='# threads for loading data')
opt_decoder.checkpoints_load_dir =  opt_decoder.parent_exp.checkpoints_dir #type=str, default='./checkpoints', help='models are saved here')
opt_decoder.checkpoints_save_dir =  opt_decoder.parent_exp.checkpoints_dir #type=str, default='./checkpoints', help='models are saved here')
opt_decoder.results_dir = opt_decoder.parent_exp.results_dir
opt_decoder.log_dir =  opt_decoder.parent_exp.log_dir #type=str, default='./checkpoints', help='models are saved here')
opt_decoder.verbose = False #action='store_true', help='if specified, print more debugging information')
opt_decoder.suffix ='' #default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')



# In[13]:
# define base_options
opt_offset_decoder = edict()

# hyperparams
# dataroot #'path to images (should have subfolders trainA, trainB, valA, valB, etc)')
opt_offset_decoder.parent_exp = opt_exp
opt_offset_decoder.batch_size = opt_offset_decoder.parent_exp.batch_size #type=int, default=1, help='input batch size')
opt_offset_decoder.ngf = 64 #type=int, default=64, help='# of gen filters in first conv layer')
opt_offset_decoder.base_model = 'resnet_decoder' #type=str, default='resnet_9blocks', help='selects model to use for netG')
opt_offset_decoder.net = 'G' #type=str, default='resnet_9blocks', help='selects model to use for netG')opt_offset_decoder.no_dropout = False #action='store_true', help='no dropout for the generator')
opt_offset_decoder.resnet_blocks = 12
opt_offset_decoder.encoder_res_blocks = 6
opt_offset_decoder.no_dropout = False
opt_offset_decoder.init_type = 'xavier' #type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
opt_offset_decoder.init_gain = 0.02 #type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
opt_offset_decoder.norm = 'instance' #type=str, default='instance', help='instance normalization or batch normalization')
opt_offset_decoder.beta1 = 0.5 #type=float, default=0.5, help='momentum term of adam')
opt_offset_decoder.lr = 0.00001 #type=float, default=0.0002, help='initial learning rate for adam')
opt_offset_decoder.lr_policy = 'step' #type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
opt_offset_decoder.lr_decay_iters = 50 #type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
opt_offset_decoder.lambda_L = 1 # weightage given to the Generator
opt_offset_decoder.lambda_cross = 0
opt_offset_decoder.lambda_reg = 0
opt_offset_decoder.weight_decay = 1e-5


opt_offset_decoder.input_nc = 4 #type=int, default=3, help='# of input image channels')
opt_offset_decoder.output_nc = 4 #type=int, default=3, help='# of output image channels')
opt_offset_decoder.save_latest_freq = 5000 #type=int, default=5000, help='frequency of saving the latest results')
opt_offset_decoder.save_epoch_freq = 1 #type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
opt_offset_decoder.n_epochs = opt_offset_decoder.parent_exp.n_epochs
opt_offset_decoder.isTrain = True
opt_offset_decoder.continue_train = False #action='store_true', help='continue training: load the latest model')
opt_offset_decoder.starting_epoch_count = opt_offset_decoder.parent_exp.starting_epoch_count #type=int, default=1, help='the starting epoch count, we save the model by <starting_epoch_count>, <starting_epoch_count>+<save_latest_freq>, ...')
opt_offset_decoder.phase = opt_offset_decoder.parent_exp.phase #type=str, default='train', help='train, val, test, etc')
opt_offset_decoder.name = 'offset_decoder' #type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
opt_offset_decoder.loss_type = "L2_offset_loss"
opt_offset_decoder.niter = 20 #type=int, default=100, help='# of iter at starting learning rate')
opt_offset_decoder.niter_decay = 100 #type=int, default=100, help='# of iter to linearly decay learning rate to zero')



opt_offset_decoder.gpu_ids = opt_offset_decoder.parent_exp.gpu_ids #type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
opt_offset_decoder.num_threads = 4 #default=4, type=int, help='# threads for loading data')
opt_offset_decoder.checkpoints_load_dir =  opt_offset_decoder.parent_exp.checkpoints_dir #type=str, default='./checkpoints', help='models are saved here')
opt_offset_decoder.checkpoints_save_dir =  opt_offset_decoder.parent_exp.checkpoints_dir #type=str, default='./checkpoints', help='models are saved here')
opt_offset_decoder.results_dir = opt_offset_decoder.parent_exp.results_dir
opt_offset_decoder.log_dir =  opt_offset_decoder.parent_exp.log_dir #type=str, default='./checkpoints', help='models are saved here')
opt_offset_decoder.verbose = False#action='store_true', help='if specified, print more debugging information')
opt_offset_decoder.suffix ='' #default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')