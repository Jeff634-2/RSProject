"""
load data
copy from lgn
"""
import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

class LastFM(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    LastFM dataset
    """
    def __init__(self, path="../data/lastfm"):
        # train or test
        cprint("loading [last fm]")
        self.mode_dict = {'train':0, "test":1}
        self.mode    = self.mode_dict['train']
        # self.n_users = 1892
        # self.m_items = 4489
        trainData = pd.read_table(join(path, 'data1.txt'), header=None)
        # print(trainData.head())
        testData  = pd.read_table(join(path, 'test1.txt'), header=None)
        # print(testData.head())
        trustNet  = pd.read_table(join(path, 'trustnetwork.txt'), header=None).to_numpy()
        # print(trustNet[:5])
        trustNet -= 1
        trainData-= 1
        testData -= 1
        self.trustNet  = trustNet
        self.trainData = trainData
        self.testData  = testData
        self.trainUser = np.array(trainData[:][0])
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = np.array(trainData[:][1])
        # self.trainDataSize = len(self.trainUser)
        self.testUser  = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem  = np.array(testData[:][1])
        self.Graph = None
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser))/self.n_users/self.m_items}")
        
        # (users,users)
        self.socialNet    = csr_matrix((np.ones(len(trustNet)), (trustNet[:,0], trustNet[:,1]) ), shape=(self.n_users,self.n_users))
        # (users,items), bipartite graph
        self.UserItemNet  = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem) ), shape=(self.n_users,self.m_items)) 
        
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        self.allNeg = []
        allItems    = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()

    @property
    def n_users(self):
        return 1892
    
    @property
    def m_items(self):
        return 4489
    
    @property
    def trainDataSize(self):
        return len(self.trainUser)
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def getSparseGraph(self):
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)
            
            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim+self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.Graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            dense = self.Graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D==0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense/D_sqrt
            dense = dense/D_sqrt.t()
            index = dense.nonzero()
            data  = dense[dense >= 1e-9]
            assert len(index) == len(data)
            self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            self.Graph = self.Graph.coalesce().to(world.device)
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1, ))
    
    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems
            
    
    
    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user
    
    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']
    
    def __len__(self):
        return len(self.trainUniqueUsers)

class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self, config = world.config, path="../data/gowalla"):
        # train or test
        cprint(f'loading [{path}]')     # world.cprint，打印数据集加载信息
        self.split = config['A_split']      # 是否分割，默认false
        self.folds = config['A_n_fold']     # 几折分割
        self.mode_dict = {'train': 0, "test": 1}    # 备注是测试还是训练
        self.mode = self.mode_dict['train']     # 0
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'      # 获取数据集路径
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0

        with open(train_file) as f:     # 打开训练集
            for l in f.readlines():     # 按行读取
                if len(l) > 0:
                    l = l.strip('\n').split(' ')    # 去除前后的换行符，然后根据空格分割
                    items = [int(i) for i in l[1:]] # 第一列是user_id，后面的是item_id
                    uid = int(l[0])                 # 提取user_id
                    trainUniqueUsers.append(uid)    # 添加到列表中
                    trainUser.extend([uid] * len(items))    # extend在表尾追加元素
                    trainItem.extend(items)                 # 表尾追加元素
                    self.m_item = max(self.m_item, max(items))  # max(items)返回列表中的最大值，此行应为max_item_id
                    self.n_user = max(self.n_user, uid)         # max_user_id
                    self.traindataSize += len(items)            # 训练集数据size扩大
        self.trainUniqueUsers = np.array(trainUniqueUsers)      # 转为数组
        self.trainUser = np.array(trainUser)                    # 转为数组
        self.trainItem = np.array(trainItem)                    # 转为数组

        with open(test_file) as f:      # 打开测试集，225-268操作同236-250
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))      # 注意这里是同名的变量，应该是用的上面那个
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1    # 这两个是保证>0？
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        
        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # 传入参数生成稀疏的csr矩阵(users,items), bipartite graph
        # data = np.ones(len(self.trainUser)); (row, col) = (self.trainUser, self.trainItem)
        # csr_matrix((data,(row,col)), shape=(,))
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()     # axis=1列维度上的求和，每一行元素相加之和; squeeze把维度中是1的去掉
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()     # axis=0行维度上求和，每一列元素相加之和
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))       # 获取积极的item
        self.__testDict = self.__build_test()
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds  # 分成多少折后，每折的长度
        for i_fold in range(self.folds):
            start = i_fold*fold_len         # 确定该折的七点
            if i_fold == self.folds - 1:    # 到了最后一折
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len   # 确定该折的终点
                # 读取为稀疏张量FloatTensor，
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)      # 转为稀疏矩阵的coo_matrix形式，且声明为float32；COO需要非0值，给非0值的行列坐标
        row = torch.Tensor(coo.row).long()      # 描述COO的行数据，转为Tensor后转为Long()
        col = torch.Tensor(coo.col).long()      # 描述COO的列数据
        index = torch.stack([row, col])         # stack堆叠，默认dim=0，在第0维度增加1维是row和col数量相加的和
        data = torch.FloatTensor(coo.data)      # 提取COO的数据转为floatTensor
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))     # COO格式的稀疏张量[位置, 值, 声明的size]


    # ===================== #
    # 获取图结构
    # ===================== #
    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:  # 还没有存在的图结构
            try:
                # --------------------- #
                # 加载拉普拉矩阵
                # --------------------- #
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz') # 从文件加载.npz格式的稀疏矩阵
                print("successfully loaded...")
                norm_adj = pre_adj_mat  # 这个是拉普拉斯矩阵
            except :
                print("generating adjacency matrix")
                s = time()

                # --------------------- #
                # 计算邻接矩阵A
                # --------------------- #
                # 创建具有初始形状(n.users+m.items, n.users+m.items)的矩阵【键字典的格式】
                # 注意这个矩阵就是拉普拉斯矩阵的A_hat，左上右下为全0，右上是原始矩阵，做下是其转置矩阵
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()   # 该矩阵转为列表格式
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R       # 填充R
                adj_mat[self.n_users:, :self.n_users] = R.T     # 填充R的转置
                adj_mat = adj_mat.todok()   # 矩阵转回键字典的格式
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])  # 后面的eye就是对对角矩阵，只要指定行数即可默认第0个对角线全为1

                # --------------------- #
                # 计算度矩阵D
                # --------------------- #
                rowsum = np.array(adj_mat.sum(axis=1))      # 列求和，然后转为array数组
                d_inv = np.power(rowsum, -0.5).flatten()    # 求数组元素的-0.5次方，然后拉伸为1维
                d_inv[np.isinf(d_inv)] = 0.                 # 无穷值处填充0
                d_mat = sp.diags(d_inv)                     # 从对角线构造一个稀疏矩阵

                # --------------------- #
                # 组合为拉普拉斯矩阵
                # --------------------- #
                norm_adj = d_mat.dot(adj_mat)               # D右乘A
                norm_adj = norm_adj.dot(d_mat)              # DA右乘D
                norm_adj = norm_adj.tocsr()                 # 返回稀疏矩阵的csr_matrix形式
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")   # 需要多少秒
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)     # 存储该稀疏矩阵

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)    # 分割加载？
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)    # 矩阵转为稀疏张量
                self.Graph = self.Graph.coalesce().to(world.device)         # coalesce对相同索引的多个值求和（压缩），再送入对应的计算设备
                print("don't split the matrix")
        return self.Graph

    # ===================== #
    # 就是构造testdict
    # ===================== #
    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):     # 在dict中get()指定的key有value，
                test_data[user].append(item)    # 把item添加到该dict的value中
            else:
                test_data[user] = [item]    # 没有了，把item转为List后填充到value中
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    # allPos = dataset.getUserPosItems(batch_users)
    # 观察到的连接作为正样本
    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])    # 网络中对应user行， nonzero()[1]获取本行中非0元素的列号
        return posItems

    # def getUserNegItems(self, users):     # 貌似这里的工作给到了CPP做
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems
