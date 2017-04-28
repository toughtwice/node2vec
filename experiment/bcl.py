#coding: UTF-8
import csv
import logging

import networkx as nx
import numpy as np
from gensim.models import Word2Vec

G=nx.Graph()
GG=nx.Graph()

def readPPIData(filepath):
    nodes_file = file(filepath+"nodes.csv", 'rb')
    reader = csv.reader(nodes_file)
    for line in reader:
        G.add_node(int(line[0]))

    edges_file = file(filepath + "edges.csv", 'rb')
    reader = csv.reader(edges_file)
    for line in reader:
        G.add_edge(int(line[0]),int(line[1]))


    # group_file=file(filepath + "groups.csv", 'rb')
    # reader = csv.reader(group_file)
    # for line in reader:
    #     G.add_edge(int(line[0]))

    G.to_undirected()
    print 'data done'
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
    print "preprocess done"
    global walks
    walks=list()
    for i in range(r):
        for node in G.nodes():
            walk=node2vecWalk(node,l)
            walks.append(walk)
    print "walks done"
    learnEmmbeding(k,d,walks)
    print "emb done"
    return

def learnEmmbeding(k,d,walks):
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=d, window=k, min_count=0, sg=1)
    model.save('/tmp/bcl_cbow_128_10_80_10_05_025.model')
    return

def loadData():
    ppipathDir = '/Users/mac/Documents/gra/BlogCatalog-dataset/data/'
    readPPIData(ppipathDir)

    return

def main():
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
    p=0.5
    q=0.25
    loadData()
    learnFeature(128,10,80,10,p,q)

if __name__ == "__main__":
	main()