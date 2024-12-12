import numpy as np
import scipy.sparse as sp
import networkx as nx
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def calculate_centrality(adj):
    degree_centrality=0
    closeness_centrality=0
    betweenness_centrality=0
    # 点度中心性
    degree_centrality = np.array(adj.sum(axis=1)).flatten()

    # 转换为 NetworkX 图对象
    G = nx.from_scipy_sparse_matrix(adj)
    
    # 接近中心性 (Closeness Centrality)
    closeness_centrality = np.array(list(nx.closeness_centrality(G).values()))

    # 中介中心性 (Betweenness Centrality)
    betweenness_centrality = np.array(list(nx.betweenness_centrality(G).values()))

    return [degree_centrality, closeness_centrality, betweenness_centrality]

def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    # 读取节点特征和标签
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    # 构建稀疏邻接矩阵
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix，将邻接矩阵转换为对称矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    print("开始获取中心性特征")

    # #获取中心性特征
    # centrality = calculate_centrality(adj)
    # #将每个特征转化为稀疏对角矩阵
    # for i in range(len(centrality)):
    #     centrality[i] = array_to_torch_sparse_diag(centrality[i])
    #
    # #保存中心性特征
    # torch.save(centrality[0], "degree_centrality_matrix.pt")
    # torch.save(centrality[1], "closeness_centrality_matrix.pt")
    # torch.save(centrality[2], "betweenness_centrality_matrix.pt")
    # print(centrality[0].shape)

    # # 读取中心性特征
    centrality=[]
    centrality.append( torch.load("degree_centrality_matrix.pt"))
    centrality.append( torch.load("closeness_centrality_matrix.pt"))
    centrality.append( torch.load("betweenness_centrality_matrix.pt"))
    print(centrality[0].shape)

    # centrality=[array_to_torch_sparse_diag(np.zeros(adj.shape[0]))]
    print("获取中心性特征完成")
    

    # 归一化特征（有归一化了）并添加自连接
    features = normalize(features)
    # 修改算法，在模型前向传播时才添加自连接
    # adj = adj + sp.eye(adj.shape[0])
    # #正则化
    # adj = normalize(adj)

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    #烦死了，直接变为稠密矩阵
    adj=adj.to_dense()
    for i in range(len(centrality)):
        centrality[i]=centrality[i].to_dense()

    return adj, features, labels, idx_train, idx_val, idx_test,centrality

# 沿行正则化，每行的和为1
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def array_to_torch_sparse_diag(array):
    """Convert a 1D numpy array to a torch sparse diagonal matrix."""
    n = len(array)
    indices = torch.arange(n, dtype=torch.int64).unsqueeze(0).repeat(2, 1)
    values = torch.from_numpy(array.astype(np.float32))
    shape = torch.Size([n, n])
    return torch.sparse.FloatTensor(indices, values, shape)
