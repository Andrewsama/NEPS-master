import networkx as nx
import numpy as np
import tensorflow as tf
import math

class DBLPDataLoader:
    def __init__(self, graph_file):
        self.g = nx.read_edgelist(graph_file, create_using=nx.Graph(), nodetype=None, data=[('weight', int)])
        self.num_of_nodes = self.g.number_of_nodes()
        self.num_of_edges = self.g.number_of_edges()
        self.edges_raw = self.g.edges()
        self.nodes_raw = self.g.nodes()


        self.embedding = []
        self.neg_nodeset = []
        self.node_index = {}
        self.node_index_reversed = {}
        for index, node in enumerate(self.nodes_raw):
            self.node_index[node] = index
            self.node_index_reversed[index] = node
        self.edges = [(self.node_index[u], self.node_index[v]) for u, v in self.edges_raw]


        ###########
        # A = nx.adjacency_matrix(self.g)
        # B = np.dot(A.T, A)
        # C = B
        # self.alias_nodes = None
        #
        # for i in np.arange(A.shape[0]):
        #     C[i, i] = 0
        #
        # self.C_ = C + A
        # self.C_ = np.array(self.C_.todense())
        ############


    def deepwalk_walk(self, walk_length):
        start_node = np.random.choice(self.nodes_raw)
        walk = [start_node]
        walk_index = [self.node_index[start_node]]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_degree = self.g.degree(cur)
            cur_nbrs = list(self.g.neighbors(cur))


            nbr_degree = [self.g.degree(nbr) for nbr in cur_nbrs]
            # for i in np.arange(len(nbr_degree)):
            #     if nbr_degree[i] == 0:
            #         nbr_degree[i] = 0.00000001

            degree = np.add(nbr_degree,cur_degree)
            probs = [1 / degree[key] for key, _ in
                     enumerate(cur_nbrs)]
            # probs = [self.C_[self.node_index[cur], self.node_index[nbr]] / degree[key] for key, nbr in
            #          enumerate(cur_nbrs)]

            norm_const = sum(probs)
            normalized_probs = [u_probs / norm_const for u_probs in probs]
            ''''''

            if len(cur_nbrs) > 0:
                walk_node = np.random.choice(cur_nbrs, p=normalized_probs)
                walk.append(walk_node)
                walk_index.append(self.node_index[walk_node])
            else:
                break
        return walk_index

    def fetch_batch(self, embedding, lu=0.1, batch_size=8, K=2, window_size=2, walk_length=8, batch = 5000 ):
        self.embedding = embedding
        self.lu = lu
        u_i = []
        u_j = []
        label = []
        embedding_dim = embedding.shape[1]
        for i in range(batch_size):
            self.walk_index = self.deepwalk_walk(walk_length)
            for index, node in enumerate(self.walk_index):
                for n in range(max(index-window_size, 0), min(index+window_size+1, walk_length)):
                    if n != index:
                        u_i.append(node)
                        u_j.append(self.walk_index[n])
                        label.append(1.)


                self.neg_nodeset = []
                u_one_hot = np.zeros(self.num_of_nodes)
                u_one_hot[node] = 1
                u_i_embedding = np.matmul(u_one_hot, self.embedding)

                temp_index = []
                # if lu >= 1 and batch > 5000:
                if lu >= 1:
                    no = self.node_index_reversed[node]
                    temp_result = []
                    no_neighbor = self.g.neighbors(no)
                    for key, value in enumerate(no_neighbor):
                        temp_result.append(value)
                        no_neighbor_neighbor = self.g.neighbors(value)
                        temp_result.extend([val for ky, val in enumerate(no_neighbor_neighbor)])

                    temp_index = [self.node_index[temp_result[i]] for i in range(len(temp_result))]

                for node_neg in self.node_index.values():
                    # if node_neg not in self.walk_index:
                    #     self.neg_nodeset.append(node_neg)

                    #if lu < 1 or batch <= 5000:
                    if lu < 1:
                        if node_neg not in self.walk_index:
                            self.neg_nodeset.append(node_neg)
                    else:
                        if node_neg not in temp_index and node_neg not in self.walk_index:
                            self.neg_nodeset.append(node_neg)

                neg_one_hot = np.zeros((len(self.neg_nodeset), self.num_of_nodes))
                for b in range(len(self.neg_nodeset)):
                    neg_one_hot[b][self.neg_nodeset[b]] = 1
                #print("neg_nodeset:",self.neg_nodeset[:10])
                negnode_embedding = np.matmul(neg_one_hot, self.embedding)
                node_negative_distribution = np.exp(np.sum(u_i_embedding * negnode_embedding, axis=1)/embedding_dim)
                #print("node_negative_distribution:",node_negative_distribution)
                node_negative_distribution /= np.sum(node_negative_distribution)

                node_negative_distribution[node_negative_distribution > lu] = 0
                #print("node_negative_distribution:",node_negative_distribution)
                neg_sum = np.sum(node_negative_distribution)
                if neg_sum != 0:
                    node_negative_distribution /= np.sum(node_negative_distribution)
                else:
                    node_negative_distribution[:] = 1/len(node_negative_distribution)

                for c in range(K):
                    # if batch <= 5000 or c <= math.ceil(K/2):
                    negative_node = np.random.choice(self.neg_nodeset, p=node_negative_distribution)
                    # else:
                    #     negative_node = np.random.choice(self.neg_nodeset)
                    u_i.append(node)
                    u_j.append(negative_node)

                    label.append(-1.)
        
        return u_i, u_j, label

    def embedding_mapping(self, embedding):
        return {node: embedding[self.node_index[node]] for node in self.nodes_raw}
    def num_node(self):
        return [self.node_index_reversed[i] for i in range(self.num_of_nodes)]

if __name__ == '__main__':
    graph_file = ''
    data_loader = DBLPDataLoader(graph_file=graph_file)
    a = np.random.rand(data_loader.num_of_nodes, 100)
    u_i, u_j, label = data_loader.fetch_batch(a,)
    print(u_i)
    print('\n---------------\n')
    print(u_j)
    print('\n---------------\n')
    print(label)





