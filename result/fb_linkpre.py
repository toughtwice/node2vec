import pickle
import random

import gensim
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics

biop=2
train_data_num=1000
def load_data():
    global G
    f2 = open("../experiment/graph.txt", "rb")
    load_list = pickle.load(f2)
    f2.close()
    G = load_list

    model_path ='/Users/mac/Documents/gra/tmp/fb_cbow_128_10_80_10_4_1.model'
    print model_path
    global model
    model = gensim.models.Word2Vec.load(model_path)
    # print G.edges(data=True)
    global nodes_num
    nodes_num=len(G.nodes())
    print "load data done"

def test():

    global logreg
    global train_negative_list
    global biop
    test_len=10000
    actual=list()
    pred=list()
    for edge in G.edges(data=True):
        if not edge[2]['samp']:
            prob=logreg.predict_proba([solve(edge[0],edge[1],biop)])#[1]   #
            # print prob
            if random.random()>(test_len+0.0)/(40000-train_data_num):
                continue
            actual.append(1)
            pred.append(prob[0][1])
    print "test positive ",len(actual)
    cnt=0
    while True:
        node1 = G.nodes()[random.randrange(nodes_num)]
        node2 = G.nodes()[random.randrange(nodes_num)]
        if not node2 in G.neighbors(node1):
            if (node1,node2) in train_negative_list:  #in train data
                continue
            prob = logreg.predict_proba([solve(node1, node2, biop)])
            cnt += 1
            actual.append(0)
            pred.append(prob[0][1])
        if cnt >= test_len:
            break
    print "test negative ",cnt

    # for node1 in G.nodes():
    #     for node2 in G.nodes():
    #         if not node2 in G.neighbors(node1):
    #             if not (edge[0], edge[1]) in train_negative_list:
    #                 if random.random() > 0.7:
    #                     continue
    #                 prob = logreg.predict_proba([solve(edge[0], edge[1], biop)])
    #                 actual.append(0)
    #                 pred.append(prob[0][1])
    print "prepare test done"
    roc_auc = metrics.roc_auc_score(actual, pred)
    print roc_auc

    fpr, tpr, thresholds = metrics.roc_curve(actual, pred)
    # roc_auc = metrics.auc(fpr, tpr)
    plt.title(roc_auc)
    plt.plot(fpr, tpr)
    plt.show()
    return



def prepare_train():
    global train_positive_list
    global train_negative_list
    global nodes_num
    train_positive_list = list()
    train_negative_list=list()
    p_list=list()
    n_list=list()
    pl = 0
    random.shuffle(G.edges(data=True))
    for edge in G.edges(data=True):
        if edge[2]['samp']:
            if random.random() > 0.2:
                pl += 1
                x=edge[0]
                y=edge[1]
                # positive_list.append((x,y,{"vec":hadamard(x,y),"exist":1}))
                train_positive_list.append((x,y))
                # p_list.append(hadamard(x,y))
                if pl>=train_data_num:
                    break

    cnt=0
    while True:
        node1=G.nodes()[random.randrange(nodes_num)]
        node2=G.nodes()[random.randrange(nodes_num)]
        if not node2 in G.neighbors(node1):
            train_negative_list.append((node1, node2))
            cnt+=1
        if cnt>=train_data_num:
            break

    print 'train positive',pl,' , negative ',cnt

    # for node1 in G.nodes():
    #     for node2 in G.nodes():
    #         if not node2 in G.neighbors(node1):
    #             if random.random() > 0.8:
    #                 # negative_list.append((node1, node2, {"vec": hadamard(x, y), "exist": 0}))
    #                 train_negative_list.append((node1, node2))
    #                 # n_list.append(hadamard(x, y))
    #             if len(train_negative_list) > train_data_num:
    #                 break
    #     if len(train_negative_list)>train_data_num:
    #         break
    # p_list.extend(n_list)
    # random.shuffle(p_list)
    nX = train_positive_list[:]
    nX.extend(train_negative_list)
    vX = [hadamard(edge[0],edge[1]) for edge in nX]
    Y = [0 if edge in train_negative_list else 1 for edge in nX]
    # print Y
    global logreg
    logreg = linear_model.LogisticRegression(penalty='l2')
    logreg.fit(vX, Y)
    print "train lr done"

def solve(x,y,para):
    if para==1:
        return average(x,y)
    if para==2:
        return hadamard(x,y)
    if para==22:
        return hadamard_2(x,y)
    if para==3:
        return weightedl1(x,y)
    if para==4:
        return weightedl2(x,y)

def average(x,y):
    global model
    vec1 = model[str(x)]
    vec2 = model[str(y)]
    prod = [(a+b+0.0)/2 for a, b in zip(vec1, vec2)]
    # print vec1
    # print vec2
    # print prod
    # print "------"
    return prod

def hadamard(x,y):
    global model
    vec1=model[str(x)]
    vec2=model[str(y)]
    prod = [a * b for a, b in zip(vec1, vec2)]
    return prod

def hadamard_2(x,y):
    global model
    vec1=model[str(x)]
    tmp1=sum([a*a for a in vec1])**0.5
    vec2=model[str(y)]
    tmp2 = sum([a * a for a in vec2]) ** 0.5
    prod = [(a * b)/(tmp1*tmp2) for a, b in zip(vec1, vec2)]
    return prod

def weightedl1(x,y):
    global model
    vec1 = model[str(x)]
    vec2 = model[str(y)]
    prod = [abs(a - b) for a, b in zip(vec1, vec2)]
    return prod

def weightedl2(x,y):
    global model
    vec1 = model[str(x)]
    vec2 = model[str(y)]
    prod = [abs(a - b)*abs(a - b) for a, b in zip(vec1, vec2)]
    return prod

def main():
    load_data()
    prepare_train()
    test()



if __name__ == "__main__":
	main()