import gensim
from sklearn.cluster import KMeans
import scipy.io as sio
import random
import sklearn.svm as svm
from sklearn.externals import joblib
import copy
import csv

train_data = list()
test_data = list()
pct = 0.8
nodes_groups_pred=dict()


clf_model=dict()
group_list=list()
nodes_groups=dict()
groups_nodes=dict()

def main():
    load_data()

    prepare_data_for_train()
    global group_list
    for label in group_list:
        prepare_classification(label)

    test()
    return

def prepare_data_for_train():
    for node in nodes_groups:
        if random.random()<pct:# or len(set(nodes_groups[node])&set([1, 4, 9, 12, 15, 18, 22, 25, 26, 27, 29, 30, 31, 32, 34, 35, 36, 38, 39]))>0:
            train_data.append(node)
        else:
            test_data.append(node)


def load_data():
    model_path='/tmp/bcl_cbow_128_10_80_10_4_1.model'
    print model_path
    global model
    model = gensim.models.Word2Vec.load('/tmp/bcl_cbow_128_10_80_10_4_1.model')
    print type(model.index2word)
    # print model['2']

    group_path = '/Users/mac/Documents/gra/BlogCatalog-dataset/data/groups.csv'
    csvfile = file(group_path, 'rb')
    reader = csv.reader(csvfile)
    global group_list
    for line in reader:
        group_list.append(int(line[0]))



    group_edges_path = '/Users/mac/Documents/gra/BlogCatalog-dataset/data/group-edges.csv'
    csvfile = file(group_edges_path, 'rb')
    reader = csv.reader(csvfile)
    global nodes_groups

    for line in reader:
        node=int(line[0])
        group=int(line[1])
        if nodes_groups.has_key(node):
            nodes_groups[node].append(group)  # node has label i
        else:
            nodes_groups[node] = list()
            nodes_groups[node].append(group)
        # if groups_nodes.has_key(group):
        #     groups_nodes[group].append(node)  # node has label i
        # else:
        #     groups_nodes[group] = list()
        #     groups_nodes[group].append(node)


#get the lr classification for label target. data includes both positive and negative
def prepare_classification(target):
    positive_list=list()
    negative_list=list()
    global nodes_groups
    for node, llist in nodes_groups.iteritems():
        if (node in train_data):
            if (target in llist):
                positive_list.append(node)
            else:
                negative_list.append(node)


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
    tp=dict.fromkeys(group_list,0)
    pre_total=dict.fromkeys(group_list,0)
    tru_total=dict.fromkeys(group_list,0)
    for node in test_data:
        global nodes_groups_pred
        nodes_groups_pred[node]=[]
        for i in group_list:
            pre_label=clf_model[i].predict([model[str(node)]])
            if (pre_label==1):
                # print "pre"
                nodes_groups_pred[node].append(i)


    for node in test_data:
        pred=nodes_groups_pred[node]
        truth=nodes_groups[node]
        print "pred:  ",pred
        print "truth:",truth
        for p_label in pred:
            if (p_label in truth):
                tp[p_label]+=1
                # print node, pred, truth
            pre_total[p_label]+=1
    for k,list in nodes_groups.iteritems():
        if k in test_data:
            for label in list:
                tru_total[label]+=1
    total_label=0
    f1=dict.fromkeys(group_list,0)
    for label in group_list:
        if pre_total[label]+tru_total[label]==0:
            continue
        if (tru_total[label]>0):
        # if tp[label] > 0:
            total_label+=1
        f1[label]=(2*tp[label]+0.0)/(pre_total[label]+tru_total[label])
    print "num of nodes:",len(nodes_groups.keys())
    print f1
    print tp
    print pre_total
    print tru_total
    # print [k if v<=0 else '' for k,v in tp.iteritems()]
    tmp=[]
    for k,v in tp.iteritems():
        if v<=0:
            tmp.append(k)
    print tmp
    print total_label
    print "micro-f1:",sum(f1.values())/total_label
    print "macro-f1:",2*(sum(tp.values())+0.0)/(sum(pre_total.values())+sum(tru_total.values()))

if __name__ == "__main__":
	main()