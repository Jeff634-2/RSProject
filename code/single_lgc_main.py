import world
import utils    # 加载utils的时候加载了try使用cpp获取负样本
from world import cprint
import torch
import numpy as np
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)      # 固定随机种子
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)  # 实际上应该是model.LightGCN(world.config, dataset)
Recmodel = Recmodel.to(world.device)                                 # 转为对应设备的代码
bpr = utils.BPRLoss(Recmodel, world.config)                          # 应该是BPRLoss的初始化，具体BPR实现在model.py中

weight_file = utils.getFileName()   # 返回了一个路径
print(f"load and save to {weight_file}")

# world.LOAD, 指令中没有声明，不重要,default=0
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")

Neg_k = 1

# init tensorboard
# if world.tensorboard:
#     w : SummaryWriter = SummaryWriter(join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment))
# else:
#     w = None
#     world.cprint("not enable tensorflowboard")
w = None

try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()

        # if epoch % 10 == 0:     # 每10个epoch输出一次，默认epoch=1000
        #     cprint("[TEST]")
        #     Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])    # 输出一次验证集信息

        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')

        # torch.save(Recmodel.state_dict(), weight_file)  # 保存模型参数

finally:
    # if world.tensorboard:
    # w.close()       # 原来W是tensorboard
    cprint("[End Epoch]")