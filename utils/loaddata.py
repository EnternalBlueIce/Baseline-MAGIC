import pickle as pkl
import time
import torch
import torch.nn.functional as F
import dgl
import networkx as nx
import json
from tqdm import tqdm
import os
from datahandlers import get_handler
from embedders import get_embedder_by_name
from partition import detect_communities

class StreamspotDataset(dgl.data.DGLDataset):
    def process(self):
        pass

    def __init__(self, name):#构造函数
        super(StreamspotDataset, self).__init__(name=name)
        if name == 'streamspot':
            path = './data/streamspot'
            num_graphs = 600
            self.graphs = []#初始化图集
            self.labels = []#初始化标签集
            print('Loading {} dataset...'.format(name))
            for i in tqdm(range(num_graphs)):#读取600张图
                idx = i
                g = dgl.from_networkx(
                    nx.node_link_graph(json.load(open('{}/{}.json'.format(path, str(idx + 1))))),
                    node_attrs=['type'],
                    edge_attrs=['type']#转换为DGL图对象并保留点类型和边类型
                )
                self.graphs.append(g)
                if 300 <= idx <= 399:
                    self.labels.append(1)#编号300到399为攻击图，其余为正常图
                else:
                    self.labels.append(0)
        else:
            raise NotImplementedError

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]#返回第i张图和标签

    def __len__(self):
        return len(self.graphs)


class WgetDataset(dgl.data.DGLDataset):
    def process(self):
        pass

    def __init__(self, name):
        super(WgetDataset, self).__init__(name=name)
        if name == 'wget':
            path = './data/wget/final'
            num_graphs = 150
            self.graphs = []
            self.labels = []
            print('Loading {} dataset...'.format(name))
            for i in tqdm(range(num_graphs)):
                idx = i
                g = dgl.from_networkx(
                    nx.node_link_graph(json.load(open('{}/{}.json'.format(path, str(idx))))),
                    node_attrs=['type'],
                    edge_attrs=['type']
                )
                self.graphs.append(g)
                if 0 <= idx <= 24:
                    self.labels.append(1)
                else:
                    self.labels.append(0)
        else:
            raise NotImplementedError

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


def load_rawdata(name):
    if name == 'streamspot':
        path = './data/streamspot'
        if os.path.exists(path + '/graphs.pkl'):#判断是否该路径的图是否已被预处理
            print('Loading processed {} dataset...'.format(name))
            raw_data = pkl.load(open(path + '/graphs.pkl', 'rb'))
        else:#否则调用构造函数
            raw_data = StreamspotDataset(name)
            pkl.dump(raw_data, open(path + '/graphs.pkl', 'wb'))
    elif name == 'wget':
        path = './data/wget'
        if os.path.exists(path + '/graphs.pkl'):
            print('Loading processed {} dataset...'.format(name))
            raw_data = pkl.load(open(path + '/graphs.pkl', 'rb'))
        else:
            raw_data = WgetDataset(name)
            pkl.dump(raw_data, open(path + '/graphs.pkl', 'wb'))
    else:
        raise NotImplementedError
    return raw_data


def load_batch_level_dataset(dataset_name):
    dataset = load_rawdata(dataset_name)#加载数据集
    graph, _ = dataset[0]
    node_feature_dim = 0
    for g, _ in dataset:
        node_feature_dim = max(node_feature_dim, g.ndata["type"].max().item())#设计特征向量维度
    edge_feature_dim = 0
    for g, _ in dataset:
        edge_feature_dim = max(edge_feature_dim, g.edata["type"].max().item())
    node_feature_dim += 1#因为编号从0开始，所以总维度要加一
    edge_feature_dim += 1
    full_dataset = [i for i in range(len(dataset))]
    train_dataset = [i for i in range(len(dataset)) if dataset[i][1] == 0]#选择训练集
    print('[n_graph, n_node_feat, n_edge_feat]: [{}, {}, {}]'.format(len(dataset), node_feature_dim, edge_feature_dim))

    return {'dataset': dataset,
            'train_index': train_dataset,
            'full_index': full_dataset,
            'n_feat': node_feature_dim,
            'e_feat': edge_feature_dim}


def transform_graph(g, node_feature_dim, edge_feature_dim):#one_hot编码
    new_g = g.clone()#克隆一份图，避免修改原图
    new_g.ndata["attr"] = F.one_hot(g.ndata["type"].view(-1), num_classes=node_feature_dim).float()
    new_g.edata["attr"] = F.one_hot(g.edata["type"].view(-1), num_classes=edge_feature_dim).float()
    return new_g


def preload_entity_level_dataset(path):#预处理节点集数据集
    path = './data/' + path
    if os.path.exists(path + '/metadata.json'):#如果包含metadata文件，则代表可能经过预处理
        pass
    else:
        print('transforming')
        train_gs = [dgl.from_networkx(
            nx.node_link_graph(g),
            node_attrs=['type'],
            edge_attrs=['type']
        ) for g in pkl.load(open(path + '/train.pkl', 'rb'))]#加载训练图
        print('transforming')
        test_gs = [dgl.from_networkx(
            nx.node_link_graph(g),
            node_attrs=['type'],
            edge_attrs=['type']
        ) for g in pkl.load(open(path + '/test.pkl', 'rb'))]#加载测试图
        malicious = pkl.load(open(path + '/malicious.pkl', 'rb'))#加载恶意节点文件
        node_feature_dim = 0
        for g in train_gs:
            node_feature_dim = max(g.ndata["type"].max().item(), node_feature_dim)
        for g in test_gs:
            node_feature_dim = max(g.ndata["type"].max().item(), node_feature_dim)
        node_feature_dim += 1
        edge_feature_dim = 0
        for g in train_gs:
            edge_feature_dim = max(g.edata["type"].max().item(), edge_feature_dim)
        for g in test_gs:
            edge_feature_dim = max(g.edata["type"].max().item(), edge_feature_dim)
        edge_feature_dim += 1
        result_test_gs = []
        for g in test_gs:
            g = transform_graph(g, node_feature_dim, edge_feature_dim)#测试图编码
            result_test_gs.append(g)#将测试图加入集合中
        result_train_gs = []
        for g in train_gs:
            g = transform_graph(g, node_feature_dim, edge_feature_dim)#训练图编码
            result_train_gs.append(g)#将训练图加入集合中
        metadata = {
            'node_feature_dim': node_feature_dim,
            'edge_feature_dim': edge_feature_dim,
            'malicious': malicious,
            'n_train': len(result_train_gs),
            'n_test': len(result_test_gs)
        }
        with open(path + '/metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f)
        for i, g in enumerate(result_train_gs):
            with open(path + '/train{}.pkl'.format(i), 'wb') as f:
                pkl.dump(g, f)
        for i, g in enumerate(result_test_gs):
            with open(path + '/test{}.pkl'.format(i), 'wb') as f:
                pkl.dump(g, f)


def load_metadata(path):#加载预处理文件
    preload_entity_level_dataset(path)
    with open('./data/' + path + '/metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return metadata


def load_entity_level_dataset(path, t, n):
    preload_entity_level_dataset(path)
    with open('./data/' + path + '/{}{}.pkl'.format(t, n), 'rb') as f:
        data = pkl.load(f)
    return data

def darpa_preload_entity_level_dataset(path):#预处理节点集数据集
    path = './data_files/theia/theia311/'
    if os.path.exists(path + '/metadata.json'):#如果包含metadata文件，则代表可能经过预处理
        pass
    else:
        print('transforming')
        train_gs = [dgl.from_networkx(
            nx.node_link_graph(g),
            node_attrs=['type'],
            edge_attrs=['type']
        ) for g in pkl.load(open(path + '/train.pkl', 'rb'))]#加载训练图
        print('transforming')
        test_gs = [dgl.from_networkx(
            nx.node_link_graph(g),
            node_attrs=['type'],
            edge_attrs=['type']
        ) for g in pkl.load(open(path + '/test.pkl', 'rb'))]#加载测试图
        malicious = pkl.load(open(path + '/malicious.pkl', 'rb'))#加载恶意节点文件
        node_feature_dim = 0
        for g in train_gs:
            node_feature_dim = max(g.ndata["type"].max().item(), node_feature_dim)
        for g in test_gs:
            node_feature_dim = max(g.ndata["type"].max().item(), node_feature_dim)
        node_feature_dim += 1
        edge_feature_dim = 0
        for g in train_gs:
            edge_feature_dim = max(g.edata["type"].max().item(), edge_feature_dim)
        for g in test_gs:
            edge_feature_dim = max(g.edata["type"].max().item(), edge_feature_dim)
        edge_feature_dim += 1
        result_test_gs = []
        for g in test_gs:
            g = transform_graph(g, node_feature_dim, edge_feature_dim)#测试图编码
            result_test_gs.append(g)#将测试图加入集合中
        result_train_gs = []
        for g in train_gs:
            g = transform_graph(g, node_feature_dim, edge_feature_dim)#训练图编码
            result_train_gs.append(g)#将训练图加入集合中
        metadata = {
            'node_feature_dim': node_feature_dim,
            'edge_feature_dim': edge_feature_dim,
            'malicious': malicious,
            'n_train': len(result_train_gs),
            'n_test': len(result_test_gs)
        }
        with open(path + '/metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f)
        for i, g in enumerate(result_train_gs):
            with open(path + '/train{}.pkl'.format(i), 'wb') as f:
                pkl.dump(g, f)
        for i, g in enumerate(result_test_gs):
            with open(path + '/test{}.pkl'.format(i), 'wb') as f:
                pkl.dump(g, f)

def darpa_load_entity_level_dataset(path, t, n):
    darpa_preload_entity_level_dataset(path)
    with open('./data_files/' + path + '/{}{}.pkl'.format(t, n), 'rb') as f:
        data = pkl.load(f)
    return data

def darpa_load_metadata(path):#加载预处理文件
    darpa_preload_entity_level_dataset(path)
    with open('./data_files/' + path + '/metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return metadata

def atlas_preload_entity_level_dataset(path):#预处理节点集数据集
    path = './atlas_data/M1'
    if os.path.exists(path + '/metadata.json'):#如果包含metadata文件，则代表可能经过预处理
        pass
    else:
        print('transforming')
        train_gs = [dgl.from_networkx(
            nx.node_link_graph(g),
            node_attrs=['type'],
            edge_attrs=['type']
        ) for g in pkl.load(open(path + '/train.pkl', 'rb'))]#加载训练图
        print('transforming')
        test_gs = [dgl.from_networkx(
            nx.node_link_graph(g),
            node_attrs=['type'],
            edge_attrs=['type']
        ) for g in pkl.load(open(path + '/test.pkl', 'rb'))]#加载测试图
        malicious = pkl.load(open(path + '/malicious.pkl', 'rb'))#加载恶意节点文件
        node_feature_dim = 0
        for g in train_gs:
            node_feature_dim = max(g.ndata["type"].max().item(), node_feature_dim)
        for g in test_gs:
            node_feature_dim = max(g.ndata["type"].max().item(), node_feature_dim)
        node_feature_dim += 1
        edge_feature_dim = 0
        for g in train_gs:
            edge_feature_dim = max(g.edata["type"].max().item(), edge_feature_dim)
        for g in test_gs:
            edge_feature_dim = max(g.edata["type"].max().item(), edge_feature_dim)
        edge_feature_dim += 1
        result_test_gs = []
        for g in test_gs:
            g = transform_graph(g, node_feature_dim, edge_feature_dim)#测试图编码
            result_test_gs.append(g)#将测试图加入集合中
        result_train_gs = []
        for g in train_gs:
            g = transform_graph(g, node_feature_dim, edge_feature_dim)#训练图编码
            result_train_gs.append(g)#将训练图加入集合中
        metadata = {
            'node_feature_dim': node_feature_dim,
            'edge_feature_dim': edge_feature_dim,
            'malicious': malicious,
            'n_train': len(result_train_gs),
            'n_test': len(result_test_gs)
        }
        with open(path + '/metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f)
        for i, g in enumerate(result_train_gs):
            with open(path + '/train{}.pkl'.format(i), 'wb') as f:
                pkl.dump(g, f)
        for i, g in enumerate(result_test_gs):
            with open(path + '/test{}.pkl'.format(i), 'wb') as f:
                pkl.dump(g, f)

def atlas_load_entity_level_dataset(path, t, n):
    atlas_preload_entity_level_dataset(path)
    with open('./atlas_data/' + path + '/{}{}.pkl'.format(t, n), 'rb') as f:
        data = pkl.load(f)
    return data

def atlas_load_metadata(path):#加载预处理文件
    atlas_preload_entity_level_dataset(path)
    with open('./atlas_data/' + path + '/metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return metadata