#coding: UTF-8
import logging
import os
import pickle

import networkx as nx
import numpy as np
import scipy
from gensim.models import Word2Vec
from scipy import io

G=nx.Graph()
GG=nx.Graph()
def readFBData(filepath):
    pathDir = os.listdir(filepath)
    for efile in pathDir:
        if efile.endswith('.edges'):
            ego=int(efile.split('.')[0])
            G.add_node(ego)
            # print ego
            child = os.path.join('%s%s' % (filepath, efile)) #文件路径
            print(child)
            # child='/Users/mac/Documents/gra/facebook/3980.edges'
            fopen = open(child, 'r')
            for line in fopen:
                x=int(line.split(' ')[0])
                y=int(line.split(' ')[1])
                G.add_node(x)
                G.add_node(y)
                G.add_edge(x,y)
                G.add_edge(ego,x,samp=1)
                G.add_edge(ego,y,samp=1)
            fopen.close()
    G.to_undirected()

    # io.savemat("G.mat",{"graph":G})
    # print "save"
    # tmp=io.loadmat("G.mat")
    # global G
    # G=tmp["graph"]
    # print "load"

    f1 = open("graph.txt", "wb")
    pickle.dump(G, f1)
    f1.close()
    f2 = open("graph.txt", "rb")
    load_list = pickle.load(f2)
    f2.close()
    # pprint.pprint(load_list.nodes())
    return

def readPPIData(filepath):
    data=scipy.io.loadmat(filepath)

    return

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
    nbrs=sorted(G.neighbors(v))
    prob=list()
    for x in nbrs:
        if (x==t):
            prob.append(1/p)
        elif x in G.neighbors(t):#x contects to t
            prob.append(1)
        else:
            prob.append(1/q)
    alias[(t,v)]=alias_solve(prob)
    return

#Return p, In-out q
def preprocessModifiedWeights(p,q):
    for edge in G.edges():
        getWeightsEdge(edge[0], edge[1],p,q)
        getWeightsEdge(edge[1], edge[0],p,q)
    return

def node2vecWalk(u,l): #从u出发走l步的list
    walk=[u]
    for i in range(l):
        curr=walk[-1]
        nbrs=sorted(G.neighbors(curr))
        if len(walk)==1:
            s=nbrs[sample(-1,curr)]
        else:
            s=nbrs[sample(walk[-2],curr)]
        walk.append(s)
    return walk

def sample(t,v):
    if t==-1: #start node.
        return int(np.floor(np.random.rand()*len(G.neighbors(v))))
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
    global walks
    walks=list()
    for i in range(r):
        for node in G.nodes():
            walk=node2vecWalk(node,l)
            walks.append(walk)
    learnEmmbeding(k,d,walks)
    return

def learnEmmbeding(k,d,walks):
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=d, window=k, min_count=0, sg=1)
    model.save('/Users/mac/Documents/gra/tmp/fb_cbow_128_10_80_10_4_1.model')
    return

def loadData():
    fbpathDir = '/Users/mac/Documents/gra/facebook/'
    readFBData(fbpathDir)

    return

def main():
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
    p=4
    q=1
    loadData()
    # learnFeature(128,10,80,10,p,q)

if __name__ == "__main__":
	main()