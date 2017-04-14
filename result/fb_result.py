import gensim
from sklearn.cluster import KMeans

def main():
    model = gensim.models.Word2Vec.load('/tmp/cbow_128_10_80_10_1_2.model')
    km=KMeans(n_cluster=3)
    km.fit(model)
    print model.most_similar('0',topn=10)
    return


if __name__ == "__main__":
	main()