#coding: UTF-8
import copy
import logging
import os
import pickle
import random

import networkx as nx
import numpy as np
from gensim.models import Word2Vec

# G=nx.Graph()
GG=nx.Graph()
remain_G=nx.Graph()
pct=0.5

def readFBData(filepath):
    # global G
    global GG
    global remain_G
    pathDir = os.listdir(filepath)
    for efile in pathDir:
        if efile.endswith('.edges'):
            ego=int(efile.split('.')[0])
            # G.add_node(ego)
            GG.add_node(ego)
            # print ego
            child = os.path.join('%s%s' % (filepath, efile)) #文件路径
            print(child)
            # child='/Users/mac/Documents/gra/facebook/3980.edges'
            fopen = open(child, 'r')
            for line in fopen:
                x=int(line.split(' ')[0])
                y=int(line.split(' ')[1])
                # G.add_node(x)
                # G.add_node(y)
                GG.add_node(x)
                GG.add_node(y)
                GG.add_edge(x,y,samp=1)
                GG.add_edge(ego,x,samp=1)
                GG.add_edge(ego,y,samp=1)
                # if random.random() < pct:
                #     G.add_edge(x, y)
                #     GG.add_edge(x, y, samp=1)
                # else:
                #     GG.add_edge(x, y, samp=0)
                # if random.random() < pct:
                #     G.add_edge(ego, x)
                #     GG.add_edge(ego, x, samp=1)
                # else:
                #     GG.add_edge(ego, x, samp=0)
                # if random.random() < pct:
                #     G.add_edge(ego, y)
                #     GG.add_edge(ego, y, samp=1)
                # else:
                #     GG.add_edge(ego, y, samp=0)
            fopen.close()
    # random.shuffle(G.nodes())
    # random.shuffle(G.edges())
    GG.to_undirected()
    print "read done"
    remain_G=copy.deepcopy(GG)
    remain_G.to_undirected()
    print "copy done"
    # io.savemat("G.mat",{"graph":G})
    # print "save"
    # tmp=io.loadmat("G.mat")
    # global G
    # G=tmp["graph"]
    # print "load"
    edge_list=GG.edges(data=True)
    samp_num=int(len(edge_list)*pct)
    random.shuffle(edge_list)
    batch=20
    for i in range(samp_num/batch):
        if i%100==0:
            print i
            print len(remain_G.nodes())
        tmp_list=[]
        for j in range(batch):
            edge=edge_list[random.randrange(samp_num)]
            x=edge[0]
            y=edge[1]
            z=edge[2]
            if GG[x][y]["samp"]==1:
                remain_G.remove_edge(x,y)
                tmp_list.append(edge)

        if connected(remain_G):
            for tmp in tmp_list:
                x = tmp[0]
                y = tmp[1]
                z = tmp[2]
                GG[x][y]["samp"] = 0
        else:
            for tmp in tmp_list:
                x = tmp[0]
                y = tmp[1]
                z = tmp[2]
                remain_G.add_edge(x, y, z)

    # G.to_undirected()
    # print len(GG.edges())
    # print len(G.edges())
    f1 = open("graph_connect_batch.txt", "wb")
    pickle.dump(GG, f1)
    f1.close()
    # f2 = open("graph.txt", "rb")
    # load_list = pickle.load(f2)
    # f2.close()
    # print load_list
    # GG = load_list
    return

def connected(g):
    len=0
    for c in nx.connected_components(g):
        len+=1
    if len>1:
        return False
    return True

def alias_solve(prob):
    small = list()
    large = list()
    length = len(prob)
    probtemp = np.zeros(length)
    aliasList = np.zeros(length, dtype=np.int)
    # probability

    for k, x in enumerate(prob):
        probtemp[k] = prob[k] * length
        if probtemp[k] < 1:
            small.append(k)  # 小于1的下标
        else:
            large.append(k)

    while (len(small) > 0 and len(large) > 0):
        ss = small.pop()
        ll = large.pop()
        aliasList[ss] = ll
        probtemp[ll] -= 1 - probtemp[ss]
        if probtemp[ll] < 1:
            small.append(ll)
        else:
            large.append(ll)
    return aliasList, probtemp

alias=dict()

#from t to v
def getWeightsEdge(t,v,p,q):
    nbrs=sorted(remain_G.neighbors(v))
    prob=list()
    for x in nbrs:
        if (x==t):
            prob.append(1/p)
        elif x in remain_G.neighbors(t):#x contects to t
            prob.append(1)
        else:
            prob.append(1/q)
    alias[(t,v)]=alias_solve(prob)
    return

#Return p, In-out q
def preprocessModifiedWeights(p,q):
    for edge in remain_G.edges():
        getWeightsEdge(edge[0], edge[1],p,q)
        getWeightsEdge(edge[1], edge[0],p,q)
    return

def node2vecWalk(u,l): #从u出发走l步的list
    walk=[u]
    for i in range(l):
        curr=walk[-1]
        nbrs=sorted(remain_G.neighbors(curr))
        if len(walk)==1:
            tmp=sample(-1, curr)
            print tmp

            s=nbrs[tmp]
        else:
            s=nbrs[sample(walk[-2],curr)]
        walk.append(s)
    return walk

def sample(t,v):
    if t==-1: #start node.
        # print "aha"
        return int(np.floor(np.random.rand()*len(remain_G.neighbors(v))))
    al,pro=alias[(t,v)]
    k=int(np.floor(np.random.rand()*len(al)))
    if np.random.rand()<pro[k]:
        return k
    else:
        return al[k]

#d 最终的feature是多少维；r 迭代多少次，l每次ranw从一个节点走几步
#k 每个节点最终选多大的context长度，Return p, In-out q
def learnFeature(d,r,l,k,p,q):
    preprocessModifiedWeights(p,q)
    print "preprocess done"
    global walks
    walks=list()
    for i in range(r):
        for node in remain_G.nodes():
            if len(remain_G.neighbors(node))<=0:
                continue
            walk=node2vecWalk(node,l)
            walks.append(walk)
    print "walks done"
    learnEmmbeding(k,d,walks)
    return

def learnEmmbeding(k,d,walks):
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=d, window=k, min_count=0, sg=1)
    model.save('/Users/mac/Documents/gra/tmp/fb_cbow_128_10_80_10_4_1_batch.model')
    return

def loadData():
    fbpathDir = '/Users/mac/Documents/gra/facebook/'
    readFBData(fbpathDir)
    print "load data done"
    return

def guaranteeConnectivity():
    for node in GG.nodes():
        conne = False
        for nei in GG.neighbors(node):
            if GG.get_edge_data(node,nei)==1:
                conne=True
                break
        if conne==False:
            return

def main():
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
    p=4
    q=1
    loadData()
    # guaranteeConnectivity()
    learnFeature(128,10,80,10,p,q)


if __name__ == "__main__":
	main()