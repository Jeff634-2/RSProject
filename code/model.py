"""
    the run of model
    copy from lgc
"""
import time

import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    # def getUsersRating(self, users):
    #     raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    # def getUsersRating(self, users):
    #     users = users.long()
    #     users_emb = self.embedding_user(users)
    #     items_emb = self.embedding_item.weight
    #     scores = torch.matmul(users_emb, items_emb.t())
    #     return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, config:dict, dataset:BasicDataset):  # 建议config为dict类型，dataset为basicdataset类型
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset.dataloader.BasicDataset = dataset  # 这里建议dataset是dataloader.BasicDataset类型的数据
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']     # 隐层的维度
        self.n_layers = self.config['lightGCN_n_layers']    # LGC的层数
        self.keep_prob = self.config['keep_prob']           # the batch size for bpr loss training procedure
        self.A_split = self.config['A_split']               # 这一项默认是在world.py中默认是false
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)   # 创建emb词数量num_users，词维度latent_dim
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:                    # whether we use pretrained weight or not
            # nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            # nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            # print('use xavier initilizer')
            # random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)        # 正态分布进行初始化权重 在torch-rechub中的initializers.py
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            # 使用预训练的Embedding
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()   # 定义sigmoid最终输出
        self.Graph = self.dataset.getSparseGraph()  # 获取离散的图结构
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # =================== #这块是什么
        # SE_block
        # self.avg_pool = nn.AdaptiveAvgPool3d(1)  # 1 * 1 * 1 的特征图
        self.channel = self.num_users+self.num_items
        self.reduction = 16
        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // self.reduction, self.channel, bias=False),
            nn.Sigmoid()
        )

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):    # (self.graph, keep_prop)
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:    # 默认这个A_split是false
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)     # 直接进入这里
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight          # 像word2vec，embedding层的权重作为user_emb和item_emb
        all_emb = torch.cat([users_emb, items_emb])     # 默认dim=0，则上下拼接两个tensor，列数不变，行数 = user_emb+item_emb
        #   torch.split(all_emb , [self.num_users, self.num_items]) # 从合并好的all_emb中再次划分出来
        embs = [all_emb]    # 转为list
        if self.config['dropout']:  # 如果设置了丢弃率
            if self.training:   # 是训练阶段
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        # 测试阶段不用dropout, 直接计算图结构   # graph在初始化权重中就构建过了
        else:
            g_droped = self.Graph    # 不然就直接计算图结构
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)   # 沿着dim=0进行拼接，第0维度为所有传入待拼接的和
                all_emb = side_emb      # 因为这里是切片加载整个图，所以把每次结果拼接后再送入中间all_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)    # sparse.mm稀疏矩阵乘法，效果同torch.mm
            embs.append(all_emb)    # emb = [all_emb=E0, E1, E2, E3]，每一个E是上下拼接的user和item，具体表示看LGC的Fig.2

        # embs_mid = self.avg_pool(embs).view(1, self.channel)
        # embs_tensor = torch.tensor(embs)
        # print(embs_tensor.shape())
        # embs_mid = self.fc(embs)
        # light_out = embs * embs_mid

        embs = torch.stack(embs, dim=1)  # 拼接，在第一维新增维度
        # # print(embs.size())
        light_out = torch.mean(embs, dim=1)  # 刚才新增的这一维变为1， 在原论文中最终的final是通过mean得到
        users, items = torch.split(light_out, [self.num_users, self.num_items])     # Ek重新分割回user和item
        return users, items

    '''
    # ========================= #
    # 在procedure.py中被调用
    # batch_users_gpu = torch.Tensor(batch_users).long().to(world.device)
    # rating = Recmodel.getUsersRating(batch_users_gpu)
    # ========================= #
    '''
    def getUsersRating(self, users):
        all_users, all_items = self.computer()      # 自己也计算了一次，因为不是被本文件中调用
        users_emb = all_users[users.long()]         # 只取本batch中的user
        items_emb = all_items                       # 和所有的item进行计算比较
        rating = self.f(torch.matmul(users_emb, items_emb.t()))     # 内积作为分数后，sigmoid输出
        return rating

    '''
    # ========================================= #
    # 计算BPRLoss的具体过程，在utils.BPRLoss中被调用 #
    # ========================================= #
    '''
    def getEmbedding(self, users, pos_items, neg_items):    # 这里转为long类型输入
        all_users, all_items = self.computer()      # 这里自己也计算了一次，因为bpr_loss的调用不在本方法中
        users_emb = all_users[users]        # 提取EuK, 因为是用的mini_batch
        pos_emb = all_items[pos_items]      # 正采样
        neg_emb = all_items[neg_items]      # 负采样
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)    # 生成了三个E0
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    # 本方法的输入应看utils/BPRLoss/stageOne
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        # reg_loss指LOSS中后面的L2正则化项
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2)) / float(len(users))

        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)

        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))    # 激活输出
        return loss, reg_loss
    '''
    # ========================================= #
    # 单次BPR计算结束 #
    # ========================================= #
    '''
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()      # 三层传播, all_users -> [Eu0, Eu1, Eu_layer]
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]        # 这里估计直接拿的是EuK和EiK
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)  # mul逐元素乘法，即做内积（矩阵乘法）
        gamma     = torch.sum(inner_pro, dim=1)      # sum(dim=1)沿着列方向求和，即每行求和得到的值放在每行的第一位
        return gamma
