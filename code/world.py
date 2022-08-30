'''
    using the parameters
    copy from LightGCN
'''

import os
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()  # 获取声明的参数

ROOT_PATH = "D:/项目/project/"
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')

import sys
sys.path.append(join(CODE_PATH, 'sources'))     # 搜素放在code_path/sources路径下的模块, 使用CPP进行模型采样

if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)   # 创建路径

config = {}
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book']
# all_models  = ['mf', 'lgn']
config['batch_size'] = 4096
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim              # the latent dimension of LGC
config['lightGCN_n_layers']= args.layer
config['dropout'] = args.dropout
config['keep_prob']  = args.keepprob                # the batch size for bpr loss training procedure与bpr_batch_size描述重复
config['A_n_fold'] = args.a_fold                    # the fold num used to split large adj matrix, like gowalla
config['test_u_batch_size'] = args.testbatch        # the batch size of users for testing
config['multicore'] = args.multicore                # whether we use multiprocessing or not in test
config['lr'] = args.lr
config['decay'] = args.decay                        # L2正则化力度
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")   # 应该是cuda 0，如果说找不到则换成cuda
CORES = multiprocessing.cpu_count() // 2            # 设置多线程的数量，前面获得是cpu总线程数，后面取一半来用
seed = args.seed                                    # 随机数的种子

dataset = args.dataset
model_name = args.model               # 使用的模型，default=lgn
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
# if model_name not in all_models:
#     raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")

TRAIN_epochs = args.epochs      # default=1000
LOAD = args.load                # default=0
PATH = args.path                # path to save weights, default="./checkpoints"
topks = eval(args.topks)        # @k test list， eval执行字符串内容
# tensorboard = args.tensorboard
comment = args.comment          # default=lgn

# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)   # simplefilter过滤器，过滤FutureWarning类型信息

def cprint(words : str):        # 打印每个包的运行信息
    print(f"\033[0;30;43m{words}\033[0m")       # 字背景颜色43，字体颜色30

# logo = r"""
# ██╗      ██████╗ ███╗   ██╗
# ██║     ██╔════╝ ████╗  ██║
# ██║     ██║  ███╗██╔██╗ ██║
# ██║     ██║   ██║██║╚██╗██║
# ███████╗╚██████╔╝██║ ╚████║
# ╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
# """
# # font: ANSI Shadow
# # refer to http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling
# # print(logo)
