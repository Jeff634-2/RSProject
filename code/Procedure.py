'''
    处理器这里有声明fit
    copy from lgn
'''
import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score


CORES = multiprocessing.cpu_count() // 2    # 多线程

'''
模型的正常运行是在这里
output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
bpr = utils.BPRLoss(Recmodel, world.config) 
'''
def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()                        # pytorch中声明为model.train()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)   # 获取一个采样的列表
    users = torch.Tensor(S[:, 0]).long()            # 第一维是用户
    posItems = torch.Tensor(S[:, 1]).long()         # 第二维是pos_item
    negItems = torch.Tensor(S[:, 2]).long()         # 第三位是negative_item

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)            # 转为对应设备的tensor

    users, posItems, negItems = utils.shuffle(users, posItems, negItems)    # 混洗
    total_batch = len(users) // world.config['bpr_batch_size'] + 1          # 计算一共分为多少个batch
    aver_loss = 0.
    for (batch_i, (batch_users, batch_pos, batch_neg)) \
            in enumerate(utils.minibatch(users, posItems, negItems, batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)   # 计算本次minibatch的BPR
        aver_loss += cri    # 累计每次的BPR
        # if world.tensorboard:
        #     w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch     # 求平均
    time_info = timer.dict()                # utils.timer.dict()    具体不知道干了什么
    timer.zero()                            # utils.timer.zero()    应该是计时器清零
    return f"loss{aver_loss:.3f}-{time_info}"
    
'''
# ============================= #
# 测试集中的一个batch
# 在下面的Test中被调用
# ============================= #
'''
# 传入是zip(rating_list, groundTrue_list)中每一组对应数据
def test_one_batch(X):
    sorted_items = X[0].numpy()     # rating_list
    groundTrue = X[1]               # 对应的groundTrue_List
    r = utils.getLabel(groundTrue, sorted_items)    # pred-data返回预测为真且实际为真的值 -> list中有多个float型的array
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)   # 计算top@K的precision和recall
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))        # 计算top@K的NDCG
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
# Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
# multicore没有在指令中声明，default=0
def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']    # the batch size of users for testing = testbatch -> default=100
    dataset: utils.BasicDataset                         # 这里只是建议类型
    testDict: dict = dataset.testDict
    Recmodel: model.LightGCN                            # 建议传入类型，只是进来这个方法的时候已经声明了
    # eval mode with no dropout
    Recmodel = Recmodel.eval()                          # eval执行，应该是用的forward？
    max_K = max(world.topks)                            # top@K
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)              # 进程池
        # ，但是默认multicore=0本if不成立

    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}      # 创建三个全0的numpy

    with torch.no_grad():       # 对本块中得到的tensor不计算梯度requires_grad=False, grad_fn=None
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10      # assert： 当本条成立时才会继续下去，否则报错

        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")

        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1        # 一共计算多少个batch
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)                       # 获取本user中非0列表，返回一个list
            groundTrue = [testDict[u] for u in batch_users]                     # 获取真值groundTrue即user真的连接

            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)                  # 转为对应设备的Tensor

            rating = Recmodel.getUsersRating(batch_users_gpu)                   # 获取本batch_user和所有item的评分
            #rating = rating.cpu()

            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))                # extend在list末尾一次性追加多个值
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)                       # 取前K个作为推荐列表输出
            rating = rating.cpu().numpy()                                   # 查看现在的rating
            # aucs = [utils.AUC(rating[i], dataset, test_data) for i, test_data in enumerate(groundTrue)]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)                          # 添加本次的所有user
            rating_list.append(rating_K.cpu())                      # 更新对应user的top@K推荐列表
            groundTrue_list.append(groundTrue)                      # 添加groundTrue
        assert total_batch == len(users_list)       # 真的遍历完所有的user才停止
        X = zip(rating_list, groundTrue_list)       # 打包预测的rating_list和真值groundTrue_list
        if multicore == 1:                          # 多线程，默认是0
            pre_results = pool.map(test_one_batch, X)
        else:                                       # 直接进到这里
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))   # 添加一个batch的实验结果
        # scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        # if world.tensorboard:   # 有开tensorboard
        #     w.add_scalars(f'Test/Recall@{world.topks}',
        #                   {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
        #     w.add_scalars(f'Test/Precision@{world.topks}',
        #                   {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
        #     w.add_scalars(f'Test/NDCG@{world.topks}',
        #                   {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:      # 程序结束，关闭线程池
            pool.close()
        print(results)
        return results
