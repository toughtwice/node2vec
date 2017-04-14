#coding: UTF-8
import os
import networkx as nx
import numpy as np
from gensim.models import Word2Vec
import logging
# import genesis.models
# import Word2Vec
import scipy.io as sio

G=nx.Graph()
GG=nx.Graph()

def readPPIData(filepath,):
    data=sio.loadmat(filepath)
    nw=data['network']
    indices=nw.indices
    indptr=nw.indptr
    for i in range(nw.indptr.shape[0]-1):
        G.add_node(i)
        for j in range(indptr[i],indptr[i + 1]):
            G.add_node(indices[j])
            G.add_edge(indices[j],i)
            # print indices[j],i
        # print indptr[i],indptr[i + 1],i
        # print indices[indptr[i]:indptr[i+1]],i,data[indptr[i]:indptr[i+1]]
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
    model.save('/tmp/ppi_cbow_128_10_80_10_4_1.model')
    return

def loadData():
    ppipathDir = '/Users/mac/Documents/gra/Homo_sapiens.mat'
    readPPIData(ppipathDir)

    return

def main():
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
    p=4
    q=1
    loadData()
    learnFeature(128,10,80,10,p,q)

if __name__ == "__main__":
	main()