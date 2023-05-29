# -*- coding:utf-8 -*-
'''
Created on 2020/5/10

@author: Andrew
'''
import numpy as np
import scipy.sparse as sp
import networkx as nx
np.random.seed(0)
dataset = 'wiki'



graph_train_file = 'data/'+dataset+'/'+dataset+'_train_edgelist.txt'
g = nx.read_edgelist(graph_train_file, create_using=nx.Graph(), nodetype=None, data=[('weight', int)])
num_of_nodes = g.number_of_nodes()
num_of_edges = g.number_of_edges()
edges_raw = g.edges()
nodes_raw = g.nodes()
print(g.nodes())

embedding = []
neg_nodeset = []
node_index = {}
node_index_reversed = {}
for index, node in enumerate(nodes_raw):
    node_index[node] = index
    node_index_reversed[index] = node
# edges = [(node_index[u], node_index[v]) for u, v in edges_raw]



graph_train_feature = 'data/'+dataset+'/'+dataset+'_train_features.txt'
train_feature = {}
with open(graph_train_feature,'r') as fn:
    x = fn.readline()
    while x:
        xx = x.strip().split()
        x_temp = [float(i) for i in xx[1:]]
        train_feature[node_index[xx[0]]] = x_temp #

        x = fn.readline()

train_feature_len = len(train_feature)



print('sasasa',np.array(train_feature[0]).shape)
train_feature_result = []
for i in range(len(train_feature)):
    train_feature_result.append(np.array(train_feature[i]))
train_feature = np.array(train_feature_result)
print('train_feature:',train_feature.shape)
# print(train_feature[0])


graph_file = 'data/'+dataset+'/'+dataset+'_edgelist.txt'

edge_list = []
original_edge_num = 0
with open(graph_file,'r') as ff:
    a = ff.readline()
    while a :
        x = a.strip().split()
        edge_list.append(node_index[x[0]])
        edge_list.append(node_index[x[1]])
        original_edge_num+=1
        a = ff.readline()
print('The number of edges is：',original_edge_num)
edges = np.array(edge_list).reshape([original_edge_num,2])



# flag = 'total'
# adj = sp.coo_matrix((np.ones(original_edge_num), (edges[:, 0], edges[:, 1])),
#                     shape=(num_of_nodes, num_of_nodes), dtype=np.float32)
# adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
# adj = np.array(adj.todense())

adj = np.zeros([num_of_nodes,num_of_nodes])
for i in range(len(edges)):
    adj[edges[i,0],edges[i,1]] = 1
    adj[edges[i,1],edges[i,0]] = 1



class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def getSimilarity(result):
    xa = np.dot(result, result.T)
    print('xa',xa.shape)
    return xa


def check_link_prediction(embedding, g, adj, check_index,N):
    def get_precisionK(embedding, g, adj, max_index, N):
        similarity = getSimilarity(embedding).reshape(-1)
        sortedInd = np.argsort(similarity)
        cur = 0
        count = 0
        temp = 0
        precisionK = []
        sortedInd = sortedInd[::-1]
        for ind in sortedInd:
            x = int(ind / N)
            y = ind % N
            if (x == y or node_index_reversed[y] in g.neighbors(node_index_reversed[x])):
                temp+=1
                continue
            count += 1
            if (adj[x][y] != 0):
                cur += 1
            # print('cur:',cur)
            # print('count:',count)
            precisionK.append(1.0 * cur / count)
            if count > max_index:
                break
        return precisionK

    precisionK = get_precisionK(embedding, g, adj, np.max(check_index),N)
    ret = []
    for index in check_index:
        print("precisonK[%d] %.5f" % (index, precisionK[index - 1]))
        ret.append(precisionK[index - 1])

    return ret


graph_train_feature = 'data/'+dataset+'/'+dataset+'_test_edgelist.txt'
num_test = 0
with open(graph_train_feature,'r') as fn:
    x = fn.readline()
    while x:
        num_test +=1
        x = fn.readline()
print('num_test:',num_test)
print()
# check_index1 = [50,100,150,200,250,300,350,400,450,500,1000,2000]
check_index_t = np.array([0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8])
check_index2 = check_index_t *num_test
check_index2 = [int(i) for i in check_index2]
# check_index2 = [int(num_test*0.6),int(num_test*0.8),num_test,int(num_test*1.2),int(num_test*1.4),int(num_test*1.6),int(num_test*1.8)]
# ret1 = check_link_prediction(train_feature, g, adj, check_index1, num_of_nodes)
ret2 = check_link_prediction(train_feature, g, adj, check_index2, num_of_nodes)
with open('data/'+dataset+'/'+dataset+'_precisionK_result.txt','w') as ff:
    # for index in range(len(check_index1)):
    #     ff.writelines("precisonK["+str(check_index1[index])+"] "+str(ret1[index])+'\n')
    for index in range(len(check_index2)):
        ff.writelines("precisonK[" + str(int(check_index_t[index]*50)) + "] " + str(ret2[index]) + '\n')
