import world
import dataloader
import model
import utils
from pprint import pprint

'''
    register注册器
    copy from lightgcn
'''

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
    dataset = dataloader.Loader(path="../data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)       # 多线程设置
print("comment:", world.comment)            # default=lgn
# print("tensorboard:", world.tensorboard)     # 是否使用tensorboard
print("LOAD:", world.LOAD)                  # default=0
print("Weight path:", world.PATH)           # path to save weights, default="./checkpoints"
print("Test Topks:", world.topks)           # tok@K
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN
}