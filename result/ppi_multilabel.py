import copy
import random

import gensim
import scipy.io as sio
import sklearn.svm as svm

nodes_labels = dict()
labels_nodes = dict()
node_label_pair = list()
train_data = list()
test_data = list()
label_list=list()
pct = 0.8
nodes_labels_pred=dict()

clf_model=dict()

def main():
    load_data()

    prepare_data_for_train()

    for label in labels_nodes:
        prepare_classification(label)

    test()
    return

def prepare_data_for_train():
    for node in nodes_labels:
        if random.random()<pct:
            train_data.append(node)
        else:
            test_data.append(node)


def load_data():

    global model
    model = gensim.models.Word2Vec.load('/tmp/ppi_cbow_128_10_80_10_4_1.model')
    print type(model.index2word)
    # print model['2']

    ppipathDir = '/Users/mac/Documents/gra/Homo_sapiens.mat'
    data = sio.loadmat(ppipathDir)
    group = data['group']
    indices = group.indices
    indptr = group.indptr
    global nodes_labels
    global labels_nodes
    global label_list
    for i in range(group.indptr.shape[0] - 1):
        for j in range(indptr[i], indptr[i + 1]):
            # print indices[j], i
            node_label_pair.append((indices[j], i))
            if not (i in label_list):
                label_list.append(i)
            node = indices[j]
            if nodes_labels.has_key(node):
                nodes_labels[node].append(i)  # node has label i
            else:
                nodes_labels[node] = list()
                nodes_labels[node].append(i)
            if labels_nodes.has_key(i):
                labels_nodes[i].append(node)
            else:
                labels_nodes[i] = list()
                labels_nodes[i].append(node)

    # print nodes_labels
    # print labels_nodes
#get the lr classification for label target. data includes both positive and negative
def prepare_classification(target):

    positive_list=list()
    negative_list=list()

    for node, llist in nodes_labels.iteritems():
        if (node in train_data):
            # print llist
            if (target in llist):
                positive_list.append(node)
            else:
                negative_list.append(node)

    positive_list.extend(positive_list)
    positive_list.extend(positive_list)

    # print len(positive_list),len(negative_list)
    # for node, label in node_label_pair:
    #     if (node in train_data):
    #         if (target==label):
    #             print target,nodes_labels[node]
    #             positive_list.append(node)
    #         else:
    #             negative_list.append(node)
    # print positive_list
    # print positive_list
    # print negative_list
    # positive_list=labels_nodes[target] #positive node of target class
    # negative_list=[]
    #
    # for k,v in labels_nodes.iteritems():
    #     if k==target:
    #         continue
    #     negative_list.extend(v)

    # length=len(positive_list) #genarate the same amount as positive
    # for i in range(int(length*1.5)):
    #     label=random.randint(max_label+1)
    #     tmp_list=labels_nodes[label]
    #     negative_list.append(tmp_list.__getitem__(random.randint(len(tmp_list))))

    positive_list.extend(negative_list)
    random.shuffle(positive_list)
    nX=positive_list[:]
    vX=[model[str(node)] for node in nX]
    Y=[0 if node in negative_list else 1 for node in nX]
    # print Y
    lin_svc=svm.LinearSVC()
    lin_svc.fit(vX, Y)
    # joblib.dump(lin_svc, "ppi/"+str(target)+"train_model.m")
    # clf_model[target]="/ppi/"+str(target)+"train_model.m"
    clf_model[target]=copy.deepcopy(lin_svc)

def test():
    tp = dict.fromkeys(label_list, 0)
    pre_total = dict.fromkeys(label_list, 0)
    tru_total = dict.fromkeys(label_list, 0)
    for node in test_data:
        global nodes_labels_pred
        nodes_labels_pred[node]=[]
        for i in label_list:
            pre_label=clf_model[i].predict([model[str(node)]])
            if (pre_label==1):
                # print "pre"
                nodes_labels_pred[node].append(i)


    for node in test_data:
        pred=nodes_labels_pred[node]
        truth=nodes_labels[node]
        print "pred:  ",pred
        print "truth:",truth
        for p_label in pred:
            if (p_label in truth):
                tp[p_label]+=1
                # print node, pred, truth
            pre_total[p_label]+=1
    for k,list in nodes_labels.iteritems():
        if k in test_data:
            for label in list:
                tru_total[label]+=1
    total_label=0
    f1=dict.fromkeys(label_list,0)
    for label in label_list:
        if pre_total[label]+tru_total[label]==0:
            continue
        #if tp[label]>0:
        if (tru_total[label] > 0):
            total_label+=1
        f1[label]=(2*tp[label]+0.0)/(pre_total[label]+tru_total[label])
    print total_label
    print f1
    print tp
    print pre_total
    print tru_total
    print "micro-f1",sum(f1.values())/total_label
    print "macro-f1:", 2 * (sum(tp.values()) + 0.0) / (sum(pre_total.values()) + sum(tru_total.values()))

if __name__ == "__main__":
	main()