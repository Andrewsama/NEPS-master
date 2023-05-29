import tensorflow as tf
import numpy as np

from model_new import NEPSModel
from utils_new import DBLPDataLoader
import pickle
import time
from sklearn.linear_model import LogisticRegression

from sklearn.manifold import TSNE
from tensorflow.python.keras import backend as K
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import os
import warnings
warnings.filterwarnings('ignor'
                        'e')


gpu_no = '0' # or '1'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no

dataset = "chameleon"
# Multiple classification test
class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return np.asarray(all_labels)


class Classifier(object):

    def __init__(self, embeddings, clf):
        self.embeddings = embeddings
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def train(self, X, Y, Y_all):
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):
        top_k_list =[len(l) for l in Y]
        self.predict(X, top_k_list)
        Y_ = self.predict(X, top_k_list)
        Y = self.binarizer.transform(Y)
        averages = ["micro", "macro"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)
        return results


    def predict(self, X, top_k_list):
        X_ = np.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X, Y, train_precent, seed=0):
        state = np.random.get_state()

        training_size = int(train_precent * len(X))
        np.random.seed(seed)
        shuffle_indices = np.random.permutation(np.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

        self.train(X_train, Y_train, Y)
        np.random.set_state(state)
        return self.evaluate(X_test, Y_test)


def read_node_label(filename, skip_head=False):
    fin = open(filename, 'r')
    X = []
    Y = []
    if skip_head:
        fin.readline()
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y


def evaluate_embeddings(embeddings):
    X, Y = read_node_label('./data/'+dataset+'/'+dataset+'_labels.txt')

    tr_frac = 0.7
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    return clf.split_train_evaluate(X, Y, tr_frac)


#training
def train(inilearning_rate=0.025, num_batches=10000, batch_size=8, K=3):
    graph_file = './data/'+dataset+'/'+dataset+'_train_edgelist.txt'
    #graph_file = './data_for_pearson/' + dataset + '/' + dataset + '_edgelist.txt'
    data_loader = DBLPDataLoader(graph_file=graph_file)
    num_of_nodes = data_loader.num_of_nodes
    num_nodes = data_loader.num_node()
    model = NEPSModel(num_of_nodes, 100, K=K)
    tt = time.ctime().replace(' ', '-')
    tt = tt.replace(':', '-')
    path = 'NEPS_new' + '-' + tt
    fout = open(path + "-log.txt", "w")

    #with tf.Session() as sess:
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        temp = 0.0

        print('batches\tloss\tsampling time\ttraining_time\tdatetime')
        tf.global_variables_initializer().run()
        a = 0.00001
        c = 0.005
        sampling_time, training_time = 0, 0
        learning_rate = inilearning_rate
        for b in range(num_batches):
            t1 = time.time()
            cur_embedding = sess.run(model.embedding)
            lu = sess.run(model.lu,feed_dict={model.a: a, model.b: c})
            u_i, u_j, label = data_loader.fetch_batch(batch_size=batch_size, K=K, embedding=cur_embedding, lu=lu,batch = b)
            feed_dict = {model.u_i: u_i, model.u_j: u_j, model.label: label,
                         model.learning_rate: learning_rate, model.a: a, model.b: c}
            t2 = time.time()
            sampling_time += t2 - t1
            if b % 50 != 0:
                sess.run(model.train_op, feed_dict=feed_dict)
                training_time += time.time() - t2

            else:
                loss = sess.run(model.loss, feed_dict=feed_dict)
                learning_rate = max(inilearning_rate * 0.0001, learning_rate*0.99)
                print('%d\t%f\t%0.2f\t%0.2f\t%s' % (b, loss, sampling_time, training_time,
                                                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                sampling_time, training_time = 0, 0
            if b % 20 == 0:
                uu = sess.run(model.u)
                print("a is %f,c is %f,lu is %f,u is %f"%(a,c,lu,uu))
            if b % 500 ==0:
                a = a * 2
            if b % 50 == 0 or b == (num_batches - 1):
                embedding = sess.run(model.embedding)
                # normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

                # fout.write("epochs:%d classification: lu=%f\n" % (b, lu))
                #
                # result = evaluate_embeddings(data_loader.embedding_mapping(normalized_embedding))
                # fout.write(str(result))
                # fout.write('\n')
                # pickle.dump(data_loader.embedding_mapping(normalized_embedding),
                #             open('data/embedding_%s%s%s5555.pkl' % (path, str(b),dataset), 'wb'))
                # pickle.dump(data_loader.embedding_mapping(normalized_embedding),
                #             open('data_for_pearson\%s\embedding_%s%s%s.pkl' % (dataset,path, str(b), dataset), 'wb'))
            fout.flush()
        with open("data/" + dataset + "/" + dataset + "_train_features.txt", 'w') as fx:
        # with open('a.txt', 'w') as fx:
            embedding = sess.run(model.embedding)
            normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            for i in range(num_of_nodes):
                fx.writelines(str(num_nodes[i])+' ')
                for j in range(len(normalized_embedding[i])):
                    fx.writelines(str(normalized_embedding[i][j]) + ' ')
                fx.writelines('\n')
    fout.close()


if __name__ == '__main__':
    train()