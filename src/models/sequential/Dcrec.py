# -*- coding: UTF-8 -*-
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset
from torch.nn.utils.rnn import pad_sequence
from typing import List
from utils import utils
from helpers.BaseReader import BaseReader
from helpers.DcrecReader import user_his
from utils.dcrec_util import TransformerLayer, TransformerEmbedding,build_sim_graph,build_adj_graph
import math
import random
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
from copy import deepcopy
from models.BaseModel import SequentialModel
import pickle
import zipfile
import pandas as pd

def cal_kl_1(target, input):
    #计算KL散度的
    target[target<1e-8] = 1e-8
    target = torch.log(target + 1e-8)
    input = torch.log_softmax(input + 1e-8, dim=0)
    return F.kl_div(input, target, reduction='batchmean', log_target=True)

class CLLayer(torch.nn.Module):
    """
    Contrastive Learning Layer是对比学习中的一个重要组成部分,是一个通用的对比学习层,
    它可以被实例化多次，分别用于计算这些不同视图对之间的对比损失(即L_u和L_v)
    """
    def __init__(self, num_hidden: int, tau: float = 0.5):
        super().__init__()
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_hidden)
        self.fc2 = torch.nn.Linear(num_hidden, num_hidden)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        """ 
        归一化计算余弦相似度
        """
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        """
        z1:查询样本q,batch_size * embedding size, 一行代表一个样本的嵌入
        z2:正样本z2,batch_size * embedding size
        该方法一次性计算整个批量的相似度矩阵(大小为[N, N]),这会导致内存复杂度为O(N²)。
        当批量大小N较大时(例如N=1000)
        ，相似度矩阵将占用大量内存(约1000x1000x4字节≈4MB),可能超出GPU内存限制。


        正样本:z1[i]与z2[i]（同一物品的序列视图和图视图）。
        负样本：包括两类：
        视图内负样本:z1[i]与z1[j](其中j ≠ i)，即同一视图下不同物品的表示。
        视图间负样本:z1[i]与z2[j](其中j ≠ i)，即不同视图下不同物品的表示。
        """
        def f(x): return torch.exp(x / self.tau)  # 温度缩放
        refl_sim = f(self.sim(z1, z1))  # z1与自身的相似度
        between_sim = f(self.sim(z1, z2))  # z1与z2的相似度,得到相似度矩阵
        #sum(1)表示按一的方向求和，即按列的方向求和,即一行（列的方向是行）求和
        return -torch.log(
            between_sim.diag()  # 正样本对的相似度,对比学习里对角线就是正样本相似度。,对角线就是由同一个物品两个表示A、B 通过A @ B^T 计算出来
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))  # 所有样本的相似度和

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        """
        
        当样本量很大时,计算完整的相似度矩阵(O(N^2))内存消耗过高。此方法将计算分批次进行，
        将大批量分成小批量
        分批次计算后合并
        """
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        def f(x): return torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def vanilla_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        def f(x): return torch.exp(x / self.tau)
        pos_pairs = f(self.sim(z1, z2)).diag()
        neg_pairs = f(self.sim(z1, z2)).sum(1)
        return -torch.log(1e-8 + pos_pairs / neg_pairs)

    def vanilla_loss_with_one_negative(self, z1: torch.Tensor, z2: torch.Tensor):
        def f(x): return torch.exp(x / self.tau)
        pos_pairs = f(self.sim(z1, z2)).diag()
        neg_pairs = f(self.sim(z1, z2))
        rand_pairs = torch.randperm(neg_pairs.size(1))
        neg_pairs = neg_pairs[torch.arange(
            0, neg_pairs.size(0)), rand_pairs] + neg_pairs.diag()
        return -torch.log(pos_pairs / neg_pairs)

    def grace_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                   mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        return ret


def graph_dual_neighbor_readout(g: dgl.DGLGraph, aug_g: dgl.DGLGraph, node_ids, features):
    """  
    这个函数用于实现用户特定consistency的计算,第2.4.1.(2)
    g:dgl.DGLGraph 完整的物品转移图，包含所有用户序列中的物品转移关系,即论文中item transition graph,用于捕获序列内物品转移模式
    aug_g:dgl.DGLGraph:增强图,去除目标用户序列边后的物品转移图,对应第2.4.1节中"perturb the item transition graph by removing edges generated from u's sequence"
    node_ids:目标节点ID,需要分析conformity程度的目标物品节点ID集合
    features:所有物品节点的嵌入表示,其中d为嵌入维度,用于邻居聚合

    输出：返回两个张量，分别代表内部邻居和外部邻居的平均嵌入。
    这些嵌入将用于计算余弦相似度,作为从众度权重的组成部分。
    本函数输出是一致性性权重计算的其中一个通道（用户特定影响），
    后续将与其他通道（如用户从众性、子图同构性）融合形成最终权重。
    """
    node_ids = node_ids.to(g.device).type(torch.int64)
    #第2.4.1.(2)节中图扰动前和后的图
    
    _, all_neighbors = g.out_edges(node_ids)#在原始物品转移图（扰动前）中,从目标节点出去的边,得到它的出边邻居
    all_nbr_num = g.out_degrees(node_ids)
    _, foreign_neighbors = aug_g.out_edges(node_ids)#在增强图（扰动后）中,从目标节点出去的边,得到它的出边邻居
    for_nbr_num = aug_g.out_degrees(node_ids)#返回tensor或int,大概率是tensor
    all_neighbors = [set(t.tolist())
                     for t in all_neighbors.split(all_nbr_num.tolist())]
    foreign_neighbors = [set(t.tolist())
                         for t in foreign_neighbors.split(for_nbr_num.tolist())]
    

    # 采样外部邻居（防止邻居过多导致计算复杂）
    for i, nbrs in enumerate(foreign_neighbors):
        if len(nbrs) > 10:
            nbrs = random.sample(nbrs, 10)
            foreign_neighbors[i] = set(nbrs)

    #civil_neighbors(内部邻居):仅在原始图中存在而不存在在增强图的邻居,代表当前用户特定行为模式
    civil_neighbors = [all_neighbors[i]-foreign_neighbors[i]
                       for i in range(len(all_neighbors))]
    #采样内部邻居
    for i, nbrs in enumerate(civil_neighbors):
        if len(nbrs) > 10:
            nbrs = random.sample(nbrs, 10)
            civil_neighbors[i] = set(nbrs)




    for_lens = [len(t) for t in foreign_neighbors]
    cv_lens = torch.tensor([len(t)
                           for t in civil_neighbors], dtype=torch.int16)
    zero_indicies = (cv_lens == 0).nonzero().view(-1).tolist()
    cv_lens = cv_lens[cv_lens > 0].tolist()

    #foreign_neighbors(外部邻居):在增强图中仍然存在的邻居，代表普遍行为模式(我们去掉了当前用户的行为对,剩下的变还在的代表的是一种普遍的行为而不是用户的特定行为)
    foreign_neighbors = torch.cat(
        [torch.tensor(list(s), dtype=torch.int64) for s in foreign_neighbors])#s for sequence,是一个set类
    civil_neighbors = torch.cat(
        [torch.tensor(list(s), dtype=torch.int64) for s in civil_neighbors])
    cv_feats = features[civil_neighbors].split(cv_lens)
    cv_feats = [t.mean(dim=0) for t in cv_feats]# 内部邻居平均池化来聚合
    # 处理内部邻居为空的情况（插入零向量）
    if len(zero_indicies) > 0:
        for i in zero_indicies:
            cv_feats.insert(i, torch.zeros_like(features[0]))
    for_feats = features[foreign_neighbors].split(for_lens)
    for_feats = [t.mean(dim=0) for t in for_feats]# 外部邻居平均池化
    return torch.stack(cv_feats, dim=0), torch.stack(for_feats, dim=0)


def graph_augment(g: dgl.DGLGraph, user_ids, user_edges):
    """
    一个图数据增强函数,用于在推荐系统中对物品关系图进行增强处理。用物品数据对图进行增强,删除共现边
    共现边指的是在用户行为序列中相邻出现的物品之间的连接关系。
     假设用户序列：[1, 3, 5, 2]
    那么会删除以下边：
    边1: 物品1 → 物品3  # 序列中第1个和第2个物品
    边2: 物品3 → 物品5  # 序列中第2个和第3个物品  
    边3: 物品5 → 物品2  # 序列中第3个和第4个物品
    假设删除的是用户A的序列边：[1,3,5,2]
    但图中还有其他用户的行为序列：
    用户B序列: [1,4,6,2] → 保留边: (1,4), (4,6), (6,2)
    用户C序列: [3,7,8,5] → 保留边: (3,7), (7,8), (8,5)
    用户D序列: [2,9,1,3] → 保留边: (2,9), (9,1), (1,3)
    g:原始的物品关系图(DGL图对象
    user_ids:当前批次的用户ID
    user_edges:存储用户历史序列中物品边关系的DataFrame
    输出:增强后的图(删除了特定的边),所谓增强不过是对数据进行处理就是增强
    """
    user_ids = user_ids.cpu().numpy()#将用户ID转换为numpy数组
    #item_edges_a和item_edges_b分别存储边的源节点和目标节点索引
    node_indicies_a = np.concatenate(
        user_edges.loc[user_ids, "item_edges_a"].to_numpy())#从user_edgesDataFrame中提取当前批次用户对应的物品边
    node_indicies_b = np.concatenate(
        user_edges.loc[user_ids, "item_edges_b"].to_numpy())
    node_indicies_a = torch.from_numpy(
        node_indicies_a).to(g.device)
    node_indicies_b = torch.from_numpy(
        node_indicies_b).to(g.device)
    node_indicies_a = node_indicies_a.type(torch.int64)
    node_indicies_b = node_indicies_b.type(torch.int64)
    edge_ids = g.edge_ids(node_indicies_a, node_indicies_b)

    aug_g: dgl.DGLGraph = deepcopy(g)
    # The features for the removed edges will be removed accordingly.
    aug_g.remove_edges(edge_ids)
    return aug_g


def graph_dropout(g: dgl.DGLGraph, keep_prob):
    # 对图的边进行dropout
    origin_edge_w = g.edata['w']

    drop_size = int((1-keep_prob) * g.num_edges())
    random_index = torch.randint(
        0, g.num_edges(), (drop_size,), device=g.device)
    mask = torch.zeros(g.num_edges(), dtype=torch.uint8,
                       device=g.device).bool()
    mask[random_index] = True
    g.edata['w'].masked_fill_(mask, 0)

    return origin_edge_w, g


class GCN(nn.Module):
    """
    实现图卷积神经网络,学习物品节点的嵌入表示
    """
    def __init__(self, in_dim, out_dim, residual = True, dropout_prob=0.3):
        """ 
        in_dim:输入节点特征的维度大小.就是物品的嵌入维度。确保图卷积层能够正确处理来自序列编码器的物品嵌入。
        out_dim:经过GCN层后输出节点嵌入的维度大小。用于后续的视图聚合。
        
        """
        super(GCN, self).__init__()
        self.dropout_prob = dropout_prob
        self.layer = GraphConv(in_dim, out_dim, weight=False,
                               bias=False, allow_zero_in_degree=False)
        self.residual = residual
        self.act = nn.GELU()
    def forward(self, graph, feature):
        """
        输入的图数据结构,可以是物品转移图v*v,物品协同图v*V。
        feature:形状为[节点数, in_dim]图中所有节点的初始特征矩阵。
        """
        graph = dgl.add_self_loop(graph)#对图加上自循环，自己到自己，在矩阵上就是加上个I


        origin_w, graph = graph_dropout(graph, 1-self.dropout_prob)
        embs = [feature]
        current_emb = feature
        for i in range(2):#两层,我们在这里可以加残差来更好实现图卷积
            current_emb = self.layer(graph, current_emb, edge_weight=graph.edata['w'])
            F.dropout(current_emb, p=0.2, training=self.training)
            self.act(current_emb)
                        # 可选残差连接
            if self.residual :
                current_emb  = current_emb  + embs[-1]
            embs.append(current_emb.clone().detach())
        embs = torch.stack(embs, dim=1)
        final_emb = torch.mean(embs, dim=1)
        graph.edata['w'] = origin_w
        return final_emb


class LightGCN(nn.Module):
    """
    基于LightGCN思想的图卷积网络，用于学习物品节点的嵌入表示
    LightGCN核心：去除非线性激活和特征变换，只保留简单的邻居聚合,邻居聚合就是邻接矩阵 * 特征矩阵，最终运算发现其实就相当自己和邻居的特征加起来，所以就是叫做聚合
    """
    def __init__(self, in_dim, out_dim, n_layers=2, dropout_prob=0.3,residual=True):
        """ 
        in_dim: 输入节点特征的维度大小
        out_dim: 输出节点嵌入的维度大小  
        n_layers: GCN层数，默认为2层
        dropout_prob: dropout概率，LightGCN中通常设置较小
        """
        super(LightGCN, self).__init__()
        self.n_layers = int(n_layers)
        self.dropout_prob = dropout_prob
        self.residual = residual
        # LightGCN不使用额外的线性变换层
        # 只保留图卷积层进行邻居聚合
        self.layers =  nn.ModuleList()
        for i in range(int(self.n_layers)):
            self.layers.append(
        GraphConv(in_dim, out_dim, weight=False, 
                                bias=False, allow_zero_in_degree=True)
            )

        
    def forward(self, graph, feature, layer_weights=None):
        """
        前向传播
        graph: DGL图，可以是物品转移图或协同交互图
        feature: 形状为[节点数, in_dim]的节点特征矩阵
        layer_weights: 各层的权重，如为None则使用均匀权重
        返回: 所有层的嵌入加权平均（LightGCN核心）

        """
        #lightGCN本身不用添加自环并在边上设置初始权重
        # graph = dgl.add_self_loop(graph)
        origin_w, graph = graph_dropout(graph,keep_prob=self.dropout_prob) 
        # 存储各层嵌入
        all_embeddings = [feature]  # 第0层是原始特征
        
        # LightGCN前向传播：简单的邻居聚合
        current_emb = feature
        for i, layer in enumerate(self.layers):
            current_emb = layer(graph, current_emb, edge_weight=graph.edata['w'])
            # 可选残差连接 
            if self.residual :
                current_emb = current_emb + all_embeddings[-1]
            all_embeddings.append(current_emb.clone().detach())
        
        # LightGCN核心：所有层嵌入的加权平均作为输出:
        # 在原始论文中，推荐使用均匀权重 1/(n_layers+1)
        all_embeddings = torch.stack(all_embeddings, dim=0)
        
        # 使用自定义层权重或均匀权重
        if layer_weights is None:
            layer_weights = torch.ones(self.n_layers + 1) / (self.n_layers + 1)
        else:
            assert len(layer_weights) == self.n_layers + 1 #权重数量应与层数匹配

            
        
        layer_weights = layer_weights.to(feature.device)
        # 对每一层应用权重并求和
        weighted_embeddings = all_embeddings * layer_weights.view(-1, 1, 1)#在最后才加权重是因为所有层权重都一样，直接最后处理
        final_emb = torch.sum(weighted_embeddings, dim=0)#求和作为输出
         # 恢复原始边权重
        graph.edata['w'] = origin_w
        return final_emb



class Dcrec(SequentialModel):
    reader,runner = 'DcrecReader','DcrecRunner'

    def _initialize_user_history(self, corpus):
        """初始化用户数据"""
        user_history = {}
        for phase in ['train', 'dev', 'test']:
            user_history[phase] = user_his(corpus, phase)
        return user_history

    def _initialize_graph_data(self, corpus):
        """初始化邻接图和交互（相似）图"""
        item_adjgraph = {}
        item_simgraph = {}
        user_edges = None
        for phase in ['train', 'dev', 'test']:
            if phase == 'train':
                item_adjgraph[phase], user_edges = build_adj_graph(self.user_history_lists,self.user_num,self.item_num,phase)
            else:
                item_adjgraph[phase], _ = build_adj_graph(self.user_history_lists,self.user_num,self.item_num,phase)
            item_simgraph[phase] = build_sim_graph(self.user_history_lists,self.user_num,self.item_num,phase)
        return item_adjgraph, item_simgraph, user_edges

    def _init_weights(self, module):
        """ 初始化权重 """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def __init__(self, args, corpus, emb_size=64, max_len=50, n_layers=2, n_heads=2, 
                 inner_size=None, dropout_rate=0.1, batch_size=512, weight_mean=0.4, 
                 kl_weight=1.0e-2, cl_lambda=1.0e-4, graph_dropout_rate=0.3):
            super().__init__(args, corpus)

            # 为 inner_size 设置默认值
            self.inner_size = inner_size if inner_size is not None else 4 * emb_size
            
            # 模型超参数
            self.device = args.device
            self.emb_size = emb_size
            self.max_len = max_len
            self.n_layers = n_layers
            self.n_heads = n_heads
            self.dropout_rate = dropout_rate
            self.batch_size = batch_size
            self.weight_mean = weight_mean
            self.kl_weight = kl_weight
            self.cl_lambda = cl_lambda
            self.graph_dropout_rate = graph_dropout_rate
            self.epochs = args.epoch
            self.current_epoch = 0
            # 嵌入层：将物品ID映射为向量,含位置编码
            self.emb_layer = TransformerEmbedding(self.item_num + 1, self.emb_size, self.max_len)

            # Transformer层堆叠
            self.transformer_layers = nn.ModuleList([
                TransformerLayer(self.emb_size, self.n_heads, self.inner_size, self.dropout_rate) 
                for _ in range(self.n_layers)
            ])

            #损失函数
            self.loss_fct = nn.CrossEntropyLoss()

            # 正则化和layer norm
            self.dropout = nn.Dropout(self.dropout_rate)
            self.layernorm = nn.LayerNorm(self.emb_size, eps=1e-12)

            #对比学习1层
            self.contrastive_learning_layer = CLLayer(self.emb_size, tau=0.8)

            # 注意力
            self.attn_weights = nn.Parameter(torch.Tensor(self.emb_size, self.emb_size))
            self.attn = nn.Parameter(torch.Tensor(1, self.emb_size))
            nn.init.normal_(self.attn, std=0.02)
            nn.init.normal_(self.attn_weights, std=0.02)

            # 初始化
            self.user_history_lists = self._initialize_user_history(corpus)
            self.item_adjgraph, self.item_simgraph, self.user_edges = self._initialize_graph_data(corpus)#这个item_simgraph对应的是物品交互图

            # 图卷积层
            self.gcn = GCN(self.emb_size, self.emb_size, residual=False,dropout_prob=self.graph_dropout_rate)

            # 权重初始化
            self.apply(self._init_weights)
    """
	Key Methods
	"""
    def _subgraph_agreement(self, aug_g, adj_graph_emb, adj_graph_emb_last_items, last_items, feed_dict, mode):
        """
        对应论文的 图中的 Multi-Channel Conformity Weighting Network,2.4.1节日
        aug_g:扰动后的图结构,形状依赖于图规模
        adj_graph_emb:邻接图(点击了一个物品后点击下一个则为1)的物品嵌入,形状为[item_num,emb_size]
        adj_graph_emb_last_items
        """
        
        
        # 通过三个语义通道计算交互级别的一致性程度：
        #视图1：用户特定影响（比较原始图与扰动图）
        # 视图2：与其他用户的一致性
        # 视图3：子图同构性
        #1.在增强图上进行GCN前向传播
        aug_output_seq = self.gcn_forward(last_items=last_items,g=aug_g)[last_items.clone().detach().type(torch.int64)]# 对应论文图中(1)user-specific conformity influence的右图的图卷积
        # 2. 双邻居读出：计算内部邻居和外部邻居的嵌入
        civil_nbr_ro, foreign_nbr_ro = graph_dual_neighbor_readout(
            self.item_adjgraph[mode], aug_g, last_items, adj_graph_emb)
        # 3. 三视图相似度计算
        view1_sim = F.cosine_similarity(
            adj_graph_emb_last_items, aug_output_seq, eps=1e-12)
        view2_sim = F.cosine_similarity(
            adj_graph_emb_last_items, foreign_nbr_ro, eps=1e-12)
        view3_sim = F.cosine_similarity(
            civil_nbr_ro, foreign_nbr_ro, eps=1e-12)
        # 4. 
        agreement = (view1_sim + view2_sim + view3_sim) / 3
        agreement = torch.sigmoid(agreement)
        agreement = (agreement - agreement.min()) / \
                    (agreement.max() - agreement.min())
        agreement = (self.weight_mean / agreement.mean()) * agreement
        return agreement
    def get_item_popularity(self, force_recompute=False):
        """获取物品流行度"""

 
        if hasattr(self, 'item_adjgraph') and self.item_adjgraph is not None:
            g = self.item_adjgraph['train']
            degrees = g.out_degrees().to(self.device) + g.in_degrees().to(self.device)
            popularity = degrees.float()

            
            # 标准化到[0,1]范围
            popularity = (popularity - popularity.min()) / (popularity.max() - popularity.min() + 1e-8)
            self._item_popularity = popularity + 1e-8
        # _item_popularity 的形状
        #_item_popularity = tensor([0.0012, 0.8543, 0.0234, 0.5678, ..., 0.0008])
        #               物品ID:   0       1       2       3    ...   item_num-1
        #               流行度: 低 ←-----------------------→ 高
        return self._item_popularity
    def _get_tail_items(self, quantile_threshold=0.8):
        """根据流行度分位数获取长尾物品"""
        item_popularity = self.get_item_popularity()
        threshold = torch.quantile(item_popularity, quantile_threshold)
        tail_items = torch.where(item_popularity < threshold)[0]
        return tail_items
    def get_attention_mask(self, item_seq, task_label=False):
        """
        生成适用于多头注意力机制的双向注意力掩码。
        item_seq:输入的项目序列
        task_label:布尔值,如果为True,会在序列开头添加标签位置
        输出:batch_size * 1 * 1 *seq_len 的四维张量,有效位置值为0,填充位置:值为-10000(在softmax后被屏蔽)
        """
        if task_label:
            #如果task_label为True,在序列开头添加一个全为1的列，用于表示任务标签
            label_pos = torch.ones((item_seq.size(0), 1), device=self.device)
            item_seq = torch.cat((label_pos, item_seq), dim=1)
        #大于0的mask为1,小于的mask为0
        attention_mask = (item_seq > 0).long()#在作为下标的时候要加long
        #将掩码从[batch_size,seq_len]扩展为[batch_size,1,1,seq_len],这是多头注意力期望的格式
        extended_attention_mask = attention_mask.unsqueeze(
            1).unsqueeze(2)  # torch.int64
        # bidirectional mask
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 
        
        #将有效位置(原来为1)转换为0,填充位置(原为0)转换为-10000,在softmax中,-10000会变成接近0的值,从而屏蔽这些位置
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def gcn_forward(self, g=None,last_items = None):
        item_emb = self.emb_layer.token_emb.weight
        item_emb = self.dropout(item_emb)
        g = g.to(item_emb.device) 
        light_out = self.gcn(g, item_emb)
        return self.layernorm(light_out + item_emb)

    def forward_loss(self, feed_dict):
        """
        使用transformer提取用户的历史记录的表示
        使用留一法，即使用历史序列预测下一个物品。
        forward_loss - 序列编码器

        专门负责序列特征提取，将用户历史序列编码为向量表示。
        """

        batch_seqs = feed_dict#batch_size * seq_len
        max_seq_len = 50   
        current_seq_len = batch_seqs.size(1)  
        if current_seq_len < max_seq_len:
            padding_len = max_seq_len - current_seq_len
            batch_seqs = F.pad(batch_seqs, (0, padding_len), value=0)
        mask = (batch_seqs > 0).unsqueeze(1).repeat(
            1, batch_seqs.size(1), 1).unsqueeze(1)
        x = self.emb_layer(batch_seqs)#batch_size * seqlen* embedding_size
        for transformer in self.transformer_layers:
            x = transformer(x, mask) #     
        return x[:, -1, :]  # [B H],只挑选最后seq  ---留一法


    def curriculum_tail_sampling(self, last_items, epoch=0, total_epochs=100):
        """课程学习：动态调整长尾样本采样策略"""
        if not self.training or total_epochs == 0:
            return last_items.clone()
        
        # 在方法内部获取tail_items，而不是作为参数传入
        tail_items = self._get_tail_items()
        if len(tail_items) == 0:
            return last_items.clone()
        
        # 动态调整采样比例
        curriculum_ratio = min(epoch / total_epochs, 1.0)
        if curriculum_ratio <=0.3:#启动时不替换
            replace_ratio = 0.0
        else:  # 
            replace_ratio = 0.15
        # 基于流行度的加权采样
        item_popularity = self.get_item_popularity()
        tail_weights = 1.0 / (item_popularity[tail_items] + 1e-8)
        tail_weights = tail_weights / tail_weights.sum()
        
        n_replace = int(last_items.shape[0] * replace_ratio)
        if n_replace == 0:
            return last_items.clone()
            
        replace_indices = torch.randperm(last_items.shape[0])[:n_replace]
        modified_last_items = last_items.clone()
        
        for idx in replace_indices:
            # 基于流行度加权采样（更偏好真正的长尾物品）
            tail_idx = torch.multinomial(tail_weights, 1)
            modified_last_items[idx] = tail_items[tail_idx]
        
        return modified_last_items
    def forward(self, feed_dict,current_epoch = 0 ,mode='test'):
        """
        整合所有组件，完成从原始输入到最终预测的完整流程。
        forward 
    ├── forward_loss (序列编码) → 序列表示
    │
    ├── gcn_forward (图编码) → 图表示
    │
    ├── curriculum_tail_sampling (长尾处理)
    │
    └── 多通道融合 → 最终预测
        """
        #利用feed_dict构建数据
        batch_user = feed_dict['user_id']#含有不同字段的字典
        batch_pos_items = feed_dict['item_id'][:, 0] 
        batch_items = feed_dict['item_id']  
        batch_seqs = feed_dict['history_items']
        # 1. 序列编码,是否考虑在序列编码上改变?
        seq_output = self.forward_loss(batch_seqs)  # 留一法,输出形状: [batch_size, emb_size]，这里使用transformer进行编码
        
        last_items = batch_seqs[:, -1].view(-1) # 形状: [batch_size],对于batch中的每个数据,即拿出最后的一个物品
        adj_graph = self.item_adjgraph[mode]
        sim_graph = self.item_simgraph[mode]

        
        # 原始形状：[item_num, emb_size]
        # iadj_graph_output_raw = tensor([[emb1],  # 物品0的嵌入
        #                                [emb2],  # 物品1的嵌入  
        #                                [emb3],  # 物品2的嵌入
        #                                ...])    # 共item_num个物品

        # # 索引数组：[batch_size]
        # last_items = tensor([2, 0, 1, 0, ...])  # 每个序列最后一个物品的ID

        # # 索引操作：iadj_graph_output_raw[last_items]
        # # 结果形状：[batch_size, emb_size]
        # result = tensor([[emb3],  # 索引2：物品2的嵌入
        #                 [emb1],  # 索引0：物品0的嵌入
        #                 [emb2],  # 索引1：物品1的嵌入
        #                 [emb1],  # 索引0：物品0的嵌入
        #                 ...])

   
        # iadj_graph_output_raw = self.gcn_forward(adj_graph)  # 形状: [item_num, emb_size], 	邻接图所有物品嵌入
        # iadj_graph_output_seq = iadj_graph_output_raw[last_items.clone().detach().type(torch.int64)]  # 形状: [batch_size, emb_size],索引后的邻接图物品嵌入
        # isim_graph_output_seq = self.gcn_forward(sim_graph)[last_items.clone().detach().type(torch.int64)]#[batch_size, emb_size],索引后的交互图物品嵌入


        #加随机采样处理长尾物品?那我就需要修改lastitem,下面的这种做法过于粗暴，直接将未出现的当作长尾物品

        # iadj_graph_output_raw = self.gcn_forward(adj_graph)  # 形状: [item_num, emb_size], 	邻接图所有物品嵌入
        # item_nums = iadj_graph_output_raw.shape[0]
        # category_last_item = torch.unique(last_items)#已经出现的
        # tail_terms_id = [i for i in range(item_nums) if i not in category_last_item]#没有出现的当作长尾物品
        # tail_terms_id = torch.tensor(tail_terms_id,dtype=torch.int64)
        # modified_lastitem_index = torch.randint(last_items.shape[0],size=(last_items.shape[0]//5,))
        # for index in modified_lastitem_index:
        #     tail_index = torch.randint(0,tail_terms_id.shape[0],size=(1,))
        #     last_items[index] = tail_index
        # 2. 课程学习：长尾物品采样（训练时启用）
        # if mode == 'train':
        #     modified_last_items = self.curriculum_tail_sampling(last_items, current_epoch, self.epochs)
        # else:
        #     modified_last_items = last_items.clone()
        #通过图卷积处理物品邻接图和相似图
        iadj_graph_output_raw = self.gcn_forward(adj_graph)  # 形状: [item_num, emb_size], 	转移图所有物品嵌入
        iadj_graph_output_seq = iadj_graph_output_raw[last_items.clone().detach().type(torch.int64)]  # 形状: [batch_size, emb_size],索引后的邻接图物品嵌入
        isim_graph_output_seq = self.gcn_forward(sim_graph)[last_items.clone().detach().type(torch.int64)]#[batch_size, emb_size],索引后的交互图物品嵌入


        # 多视图注意力融合,figure2左边的
        mixed_x = torch.stack((seq_output, iadj_graph_output_seq, isim_graph_output_seq), dim=0)
        # mixed_x形状: [3, batch_size, emb_size]
        weights = (torch.matmul(mixed_x, self.attn_weights.unsqueeze(0)) * self.attn).sum(-1)#乘以注意力权重
        score = F.softmax(weights, dim=0).unsqueeze(-1)
        seq_output = (mixed_x * score).sum(0)
        item_indices = batch_items.view(-1)  
        test_item_emb = self.emb_layer.token_emb(item_indices)  # 形状: [batch_size*num_items, emb_size]
        batch_size, num_items = batch_items.size()
        test_item_emb = test_item_emb.view(batch_size, num_items, -1)# 形状: [batch_size, num_items, emb_size]
        seq_output = seq_output.unsqueeze(1)
        scores = torch.bmm(seq_output, test_item_emb.transpose(1, 2)).squeeze(1)
        # scores形状: [batch_size, num_items]
        return {'prediction': scores}



    def loss(self, feed_dict, mode='train'):
        """
        输入:训练数据字典和模式
        输出:总损失和损失明细字典
        """
    
        batch_user = feed_dict['user_id']#当前批次用户
        batch_pos_items = feed_dict['item_id'][:, 0]#正样本物品
        batch_seqs = feed_dict['history_items']#用户历史序列
        last_items = batch_seqs[:, -1].view(-1)#序列最后一个物品
        #在loss出不加课程学习,因为不是在训练的时候
        # 图数据处理
        masked_g = self.item_adjgraph[mode]#原始邻接图
        aug_g = graph_augment(self.item_adjgraph[mode], batch_user, self.user_edges)#增强后的图
        adj_graph_emb = self.gcn_forward(masked_g)#对邻接图进行图卷积，对应原论文里的collaborative signals 指向 Graph Convolution
        sim_graph_emb = self.gcn_forward(self.item_simgraph[mode])#相似图嵌入
        adj_graph_emb_last_items = adj_graph_emb[last_items.clone().detach().type(torch.int64)]
        sim_graph_emb_last_items = sim_graph_emb[last_items.clone().detach().type(torch.int64)]

        seq_output = self.forward_loss(batch_seqs)#主序列编码,考虑在序列编码上改变?
        aug_seq_output = self.forward_loss(batch_seqs)#增强序列编码

        # 三通道计算CL weights
        mainstream_weights = self._subgraph_agreement(
            aug_g, adj_graph_emb, adj_graph_emb_last_items, last_items, feed_dict, mode)
        # 过滤 len=1, set weight=0.5
        seq_lens = batch_seqs.ne(0).sum(dim=1)
        #序列长度过滤和KL散度正则化
        mainstream_weights[seq_lens == 1] = 0.5

        expected_weights_distribution = torch.normal(self.weight_mean, 0.1, size=mainstream_weights.size()).to(
            self.device)
        kl_loss = self.kl_weight * cal_kl_1(expected_weights_distribution.sort()[0], mainstream_weights.sort()[0])

        personlization_weights = mainstream_weights.max() - mainstream_weights

        # 对比学习
        cl_loss_adj = self.contrastive_learning_layer.vanilla_loss(
            aug_seq_output, adj_graph_emb_last_items)#序列vs转移图对比损失
        #转移图vs图卷积图对比损失
        cl_loss_a2s = self.contrastive_learning_layer.vanilla_loss(
            adj_graph_emb_last_items, sim_graph_emb_last_items)
        #加权融合对比损失
        cl_loss = (self.cl_lambda * (mainstream_weights *
                                     cl_loss_adj + personlization_weights * cl_loss_a2s)).mean()
        # 融合
        # 3, N_mask, dim
        mixed_x = torch.stack(
            (seq_output, adj_graph_emb[last_items.clone().detach().type(torch.int64)], sim_graph_emb[last_items.clone().detach().type(torch.int64)]), dim=0)
        weights = (torch.matmul(
            mixed_x, self.attn_weights.unsqueeze(0)) * self.attn).sum(-1)
        # 3, N_mask, 1
        score = F.softmax(weights, dim=0).unsqueeze(-1)
        seq_output = (mixed_x * score).sum(0)
        # [item_num, H]
        test_item_emb = self.emb_layer.token_emb.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        batch_pos_items = batch_pos_items.to(torch.int64)
        loss = self.loss_fct(logits + 1e-8, batch_pos_items) #1.主损失

        loss_dict = {
            "loss": loss.item(),
            "cl_loss": cl_loss.item(),
            "kl_loss": kl_loss.item(),
        }
        return loss + cl_loss + kl_loss, loss_dict
