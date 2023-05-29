# -*- coding:utf-8 -*-
'''
Created on 2020/5/8

@author: Andrew
'''
import numpy as np
# np.random.seed(0)
edge = []
dataset = 'email-Eu-core'
la = []
node_count = {}
# with open(dataset+'/'+dataset+'_labels.txt','r') as ff:
#     xx = ff.readline()
#     while xx:
#         xx = xx.strip().split()
#         node_count[xx[0]] = 0
#         xx = ff.readline()

with open(dataset+'/'+dataset+'_edgelist.txt','r') as fn:
    x = fn.readline()
    while x:
        xx = x.strip().split()
        edge.append(xx)
        la.append(xx[0])
        la.append(xx[1])
        if xx[0] not in node_count.keys():
            node_count[xx[0]] = 1
        else:
            node_count[xx[0]] += 1
        if xx[1] not in node_count.keys():
            node_count[xx[1]] = 1
        else:
            node_count[xx[1]] += 1

        # node_count[xx[0]] += 1
        # node_count[xx[1]] += 1
        x = fn.readline()
# idx_labels = np.genfromtxt(dataset+"/"+dataset+"_labels.txt", dtype=np.dtype(str))
# num_ = len(idx_labels[:,0])
num_edge = set(la)
# print(num_edge)
num_ = len(num_edge)
edge_len = len(edge)
print(edge_len)
edge_test = int(edge_len * 0.5)
print(edge_test)
edge_train = edge_len - edge_test
a = range(edge_len)
print(len(a))

def choose():
    # edge_test_index = np.random.choice(a,size=edge_test,repllace=False)
    # # print(len(edge_test_index))
    # edge_train_index = np.delete(a,edge_test_index)
    # # print(len(edge_train_index))
    # x1 = []
    # for i in edge_train_index:
    #     x1.append(int(edge[i][0]))
    #     x1.append(int(edge[i][1]))
    # x1 = set(x1)
    # num_node = num_- len(x1)
    count = 0
    bb = range(edge_len)
    x1 = []
    while(1):

        edge_train_index = np.random.choice(bb)
        if (node_count[edge[edge_train_index][0]] == 1) or (node_count[edge[edge_train_index][1]] == 1):
            continue
        if edge[edge_train_index][0] == edge[edge_train_index][1] and node_count[edge[edge_train_index][0]] == 2:
            continue
        node_count[edge[edge_train_index][0]] -= 1
        node_count[edge[edge_train_index][1]] -= 1
        delete_index = np.where(bb == edge_train_index)[0][0]
        print('The deleted index is',delete_index)
        bb = np.delete(bb, delete_index)
        count+=1
        if count == edge_test:
            break
    for i in bb:
        x1.append(int(edge[i][0]))
        x1.append(int(edge[i][1]))
    x1 = set(x1)
    num_node = num_ - len(x1)

    edge_test_index = np.delete(a,bb)
    return x1,bb,edge_test_index,num_node

tor = 0

n=0
x1,bb,edge_test_index,num_node = choose()
if len(x1) != num_:
    print('raise error')
else:
    print('ok')
    with open(dataset + '/' + dataset + '_train_edgelist.txt', 'w') as ff:
        for i in bb:
            ff.writelines(edge[i][0]+' '+edge[i][1]+'\n')

    with open(dataset+'/'+dataset+'_test_edgelist.txt','w') as ff:
        for i in edge_test_index:
            ff.writelines(edge[i][0]+' '+edge[i][1]+'\n')
# while(1):
#     x1,edge_train_index,edge_test_index,num_node = choose()
#     if len(x1) != num_:
#         tor += 1
#         if tor %10 == 0:
#             print(tor)
#         if tor > 10000:
#             break
#         continue
#     else:
#         print("complete，the tor is：",tor)
#
#
#         # with open(dataset+'/'+dataset+'_train_edgelist.txt','w') as ff:
#         #     for i in edge_train_index:
#         #         ff.writelines(edge[i][0]+' '+edge[i][1]+'\n')
#         #
#         # with open(dataset+'/'+dataset+'_test_edgelist.txt','w') as ff:
#         #     for i in edge_test_index:
#         #         ff.writelines(edge[i][0]+' '+edge[i][1]+'\n')
#     break






#
# x1,edge_train_index,edge_test_index = choose()
# with open(dataset+'/'+dataset+'_train_edgelist.txt','w') as ff:
#     for i in edge_train_index:
#         ff.writelines(edge[i][0]+' '+edge[i][1]+'\n')
#
# with open(dataset+'/'+dataset+'_test_edgelist.txt','w') as ff:
#     for i in edge_test_index:
#         ff.writelines(edge[i][0]+' '+edge[i][1]+'\n')
# print("sole")




















#
# import numpy as np
# import scipy.sparse as sp
# import torch
# def encode_onehot(labels):
#     classes = set(labels)
#     classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
#     labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
#     return labels_onehot
# def load_data(path="../data/cora/", dataset="cora"):
#     """Load citation network dataset (cora only for now)"""
#     print('Loading {} dataset...'.format(dataset))
#     idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
#     features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
#     labels = encode_onehot(idx_features_labels[:, -1])
#     # build graph
#     idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
#     idx_map = {j: i for i, j in enumerate(idx)}
#     edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
#     edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
#     adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
#                         shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
#     # build symmetric adjacency matrix
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#     features = normalize(features)
#     adj = normalize(adj + sp.eye(adj.shape[0]))
#     idx_train = range(140)
#     idx_val = range(200, 500)
#     idx_test = range(500, 1500) f
#     eatures = torch.FloatTensor(np.array(features.todense()))
#     labels = torch.LongTensor(np.where(labels)[1])
#     adj = sparse_mx_to_torch_sparse_tensor(adj)
#     idx_train = torch.LongTensor(idx_train)
#     idx_val = torch.LongTensor(idx_val)
#     idx_test = torch.LongTensoridx_test)
#     return adj, features, labels, idx_train, idx_val, idx_test
#
#






















