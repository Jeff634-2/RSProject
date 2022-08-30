'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import world
import torch
from torch import nn, optim
import numpy as np
from torch import log
from dataloader import BasicDataset
from time import time
from model import LightGCN
from model import PairWiseModel
from sklearn.metrics import roc_auc_score
import random
import os

try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    path = join(dirname(__file__), "sources/sampling.cpp")
    sampling = imp_from_filepath(path)
    sampling.seed(world.seed)
    sample_ext = True
except:
    world.cprint("Cpp extension not loaded")
    sample_ext = False

# main中有初始化 bpr = utils.BPRLoss(Recmodel, world.config)
class BPRLoss:
    def __init__(self, recmodel : PairWiseModel, config : dict):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)   # 注意这里是回到了model.py中的bpr_loss计算单次loss
        reg_loss = reg_loss*self.weight_decay       # reg_loss指L2正则化项
        loss = loss + reg_loss

        self.opt.zero_grad()    # 参数初始化为0
        loss.backward()         # 反向传播计算损失
        self.opt.step()         # 更新所有参数

        # loss.cpu().item() 把数值提取出来
        return loss.cpu().item()

# S = utils.UniformSample_original(dataset)
# dataset = dataloader.Loader(path="../data/"+world.dataset)
def UniformSample_original(dataset, neg_ratio = 1):
    dataset : BasicDataset
    allPos = dataset.allPos
    start = time()
    if sample_ext:      # 在导入utils使用CPP的阶段，若使用CPP成功则sample_ext=true
        S = sampling.sample_negative(dataset.n_users, dataset.m_items,
                                     dataset.trainDataSize, allPos, neg_ratio)
    else:
        S = UniformSample_original_python(dataset)      # 也就是CPP不是必须要执行的，只是CPP可以加速采样过程
    return S

def UniformSample_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time()
    dataset : BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)         # randint返回一个随机整型，范围[0, dataset.n_users)，随机数的尺寸为user_num
    allPos = dataset.allPos
    S = []      # 采样结果
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:            # 用户没有交互数据？不处理
            continue
        sample_time2 += time() - start      # 记录计算所用时间
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]                      # 获取一个正样本
        while True:
            negitem = np.random.randint(0, dataset.m_items)     # 随机生成
            if negitem in posForUser:       # 该随机获取的样本在正样本序列中
                continue
            else:
                break
        S.append([user, positem, negitem])      # 组装返回的序列
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S)

# ===================end samplers==========================

# =====================utils====================================

# utils.set_seed(world.seed)
def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)        # 固定GPU的随机生成种子
        torch.cuda.manual_seed_all(seed)    # 为所有的GPU固定种子
    torch.manual_seed(seed)

# weight_file = utils.getFileName()
def getFileName():
    if world.model_name == 'mf':
        file = f"mf-{world.dataset}-{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name == 'lgn':     # 进入这里
        file = f"lgn-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}.pth.tar"
    return os.path.join(world.FILE_PATH, file)

# * 组成tuple, ** 组成dict
# utils.minibatch(users, batch_size=u_batch_size)
def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])   # 从可变参数中获取batch_size

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]      # yield返回一个迭代器，类似return，但是每次返回本for的结果不退出循环
    else:
        for i in range(0, len(tensors[0]), batch_size):     # range声明起点，终点，步长
            yield tuple(x[i:i + batch_size] for x in tensors)

# users, posItems, negItems = utils.shuffle(users, posItems, negItems)
# * 接受一个tuple， ** 接受一个dict
def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)      # 在dict类型的可变参数中获取indices, default=false

    if len(set(len(x) for x in arrays)) != 1:       # 这里应该指获取user的长度？
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))     # 组装为一个范围
    np.random.shuffle(shuffle_indices)              # 在范围内混洗

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:     # default=false
        return result, shuffle_indices
    else:
        return result


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


# ====================Metrics==============================
# =========================================================
'''
计算计算top@K的precision和recall
在procedure/test_one_batch中使用
'''
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    # 传入R Ru和Tu的交集 Ru 我们给出的推荐列表，Tu用户的实际交互列表
    right_pred = r[:, :k].sum(1)    # 所有行=所有用户，前K列=topK, 这里sum(1)应该是计算总和后+1保证不为0 tp
    precis_n = k                    # 获取top@K
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])    # 全集Tu：用户实际选择的物品集
    recall = np.sum(right_pred/recall_n)    # recall = tp / (tp+fn)
    precis = np.sum(right_pred)/precis_n    # precision = tp / (tp+fp)
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

'''
计算计算top@K的NDCG归一化折损累计增益
在procedure/test_one_batch中使用
'''
def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset : BasicDataset
    r_all = np.zeros((dataset.m_items, ))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)

'''
# ======================================== #
# 在procedure/test/test_one_epoch中被调用，获取一个zip中的标签组
r = utils.getLabel(groundTrue, sorted_items)
# ======================================== #
'''
def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]  # 提取本次推荐中的所有真实值和预测值
        pred = list(map(lambda x: x in groundTrue, predictTopK))    # map做映射，同时存在于两个列表中，则表示命中，预测为真映射输出
        pred = np.array(pred).astype("float")   # 真序列转为array数组后转为float型数组
        r.append(pred)  # 本用户的预测真值返回
    return np.array(r).astype('float')

# ====================end Metrics=============================
# =========================================================
