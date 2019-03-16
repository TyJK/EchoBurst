import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from mpl_toolkits.mplot3d import Axes3D
from logging import basicConfig, INFO
from random import shuffle
from sklearn import svm
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA



class LabeledLineSentence(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        deck = []
        for line in open(self.filename, encoding="utf-8"):
            deck.append(line)
            if len(deck) >= 10000000:
                shuffle(deck)
                for card in deck:
                    csv = card.split(",")
                    subreddit = csv[0]
                    body = csv[1].split()
                    yield TaggedDocument(words=body, tags=[subreddit, clusterLabel[subreddit]])
                deck = []


def trainNewModel(inputFile, outputFile, model):
    documents = LabeledLineSentence("Data\\" + inputFile)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count)
    model.save("Models\\" + outputFile)


def retrainModel(vectorFile, dataFile, outputFile, iterations):
    documents = LabeledLineSentence("Data\\" + dataFile)
    model = Doc2Vec.load("Models\\" + vectorFile)
    for epoch in range(iterations):
        model.train(documents)
    model.save("Models\\" + outputFile)


def testModel(inputFile):
    model = Doc2Vec.load("Models\\" + inputFile)
    while True:
        choice = input("Press 1 to compare documents within the model to each other.\n"
                       "Press 2 to run similarity tests on individual words.\n"
                       "Press 3 to get the top related subreddits for an inferred new vector (comment).\n"
                       "Hit any key to exit.\n")
        if choice == "1":
            docChoice = input("Enter the subreddit you want to test.\n")
            print(model.docvecs.most_similar(docChoice))
        elif choice == "2":
            wordChoice = input("Enter the word you wish to analyze.\n").lower()
            print(model.most_similar(wordChoice))
        elif choice == "3":
            with open("testing.txt") as t:
                resultList = []
                testDocs = t.readlines()
                for doc in testDocs:
                    doc = doc.split("\t")
                    tag = doc[0]
                    body = doc[1]
                    newVec = model.infer_vector(body.split())
                    resultList.append("The original category is {}: {}\n {}\n".
                                      format(tag, body, model.docvecs.most_similar(positive=[newVec])))
                with open("clusteredResults.txt", "a") as x:
                    for element in resultList:
                        x.write(element)
        else:
            break


def newKMeansModel(vectorFile, outputFile, numClusters):
    # https://stackoverflow.com/questions/43476869/doc2vec-sentence-clustering

    model = Doc2Vec.load("Models\\" + vectorFile)
    docVecs = model.docvecs.doctag_syn0
    km = KMeans(n_clusters=numClusters)
    print("Starting")
    km.fit(docVecs)
    print("Fitting Data")
    joblib.dump(km, outputFile)


def loadKMeansModel(vectorFile, clusterFile, csvFile):
    # https://stackoverflow.com/questions/43476869/doc2vec-sentence-clustering

    model = Doc2Vec.load("Models\\" + vectorFile)
    km = joblib.load(clusterFile)
    clusters = km.labels_.tolist()
    cluster_info = {'labels': model.docvecs.offset2doctag,
                    "index, wordcount and repeated words": [model.docvecs.doctags[x] for x in model.docvecs.offset2doctag],
                    'clusters': clusters}
    sentenceDF = pd.DataFrame(cluster_info, index=[clusters],
                              columns=['labels', "index, wordcount and repeated words", 'clusters'])
    print(sentenceDF)
    sentenceDF.to_csv(csvFile)


def newDBSCANModel(vectorFile, outputFile):
    model = Doc2Vec.load("Models\\" + vectorFile)
    vecs = []
    for doc in range(0, len(model.docvecs)):
        doc_vec = model.docvecs[doc]
        # print doc_vec
        vecs.append(doc_vec.reshape((1, 300)))

    doc_vecs = np.array(vecs, dtype='float')  # TSNE expects float type values

    # print doc_vecs
    docs = []
    for i in doc_vecs:
        docs.append(i[0])
    db = DBSCAN(eps=0.03, algorithm="brute", metric='cosine').fit(docs)
    joblib.dump(db, outputFile)


    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    clusters = db.labels_.tolist()
    cluster_info = {'labels': model.docvecs.offset2doctag,
                    "index, wordcount and repeated words": [model.docvecs.doctags[x] for x in
                                                            model.docvecs.offset2doctag],
                    'clusters': clusters}
    sentenceDF = pd.DataFrame(cluster_info, index=[clusters],
                              columns=['labels', "index, wordcount and repeated words", 'clusters'])
    print(sentenceDF)
    sentenceDF.to_csv("DBSCAN.csv")

    print('Estimated number of clusters: %d' % n_clusters_)


def plotModel2D(vectorFile, numClusters):
    # http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html

    model = Doc2Vec.load("Models\\" + vectorFile)
    docVecs = model.docvecs.doctag_syn0
    reduced_data = PCA(n_components=10).fit_transform(docVecs)
    kmeans = KMeans(init='k-means++', n_clusters=numClusters, n_init=10)
    kmeans.fit(reduced_data)
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap="hot",
               aspect='auto', origin='lower')
    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)

    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on Reddit Text Data(PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def plotModel3D(vectorFile, numClusters):
    # http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html

    model = Doc2Vec.load("Models\\" + vectorFile)
    docVecs = model.docvecs.doctag_syn0
    reduced_data = PCA(n_components=10).fit_transform(docVecs)
    kmeans = KMeans(init='k-means++', n_clusters=numClusters, n_init=10)

    fig = plt.figure(1, figsize=(10, 10))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    kmeans.fit(reduced_data)
    labels = kmeans.labels_

    ax.scatter(reduced_data[:, 5], reduced_data[:, 2], reduced_data[:, 3], c=labels.astype(np.float))
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    # Plot the ground truth
    fig = plt.figure(1, figsize=(10, 10))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    plt.cla()
    ax.scatter(reduced_data[:, 5], reduced_data[:, 2], reduced_data[:, 3], c=labels.astype(np.float))
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    plt.show()


def clusterLabeler(csvFile):
    with open(csvFile, 'r') as csvfile:
        read = csv.reader(csvfile, delimiter=',')
        mydict = {rows[1]: rows[0] for rows in read}

        return mydict


basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=INFO)

# Use to create a dictionary of subreddits and their respective clusters. If not using, remove the 'clusterLabel' tag in LabeledLineSentence
# clusterLabel = clusterLabeler("clusters.csv")

# Train a new model from scratch based on your own corpus. Depending on your CPU you may need to change the number of workers
# trainNewModel("miniFeb2017SubredditIncluded.txt", "yourOutputFile", Doc2Vec(dm=0, iter=20, dm_mean=1, size=300, window=5, negative=5, min_count=10, workers=7))

# Retrain a model for some number of iterations, using the same vocabulary as before.
# retrainModel("Doc2VecFeb2017MiniCorpus", "miniFeb2017SubredditIncluded.txt", "Doc2VecFeb2017UpdatedMiniCorpus", 3)

# Run testing suit on model
# testModel("clusteredModel")

# Create a new KMeans model, and save it
# newKMeansModel("Doc2VecFeb2017MiniCorpus", "KMeans_cluster.pkl", 40)

# Load a KMeans model and export the resulting clusters to a csv, along with subreddit name and other information
# loadKMeansModel("Doc2VecFeb2017MiniCorpus", "KMeans_cluster.pkl", "clusters.csv")

# Use demntionality reduction to plot KMeans clusters
# plotModel2D("Doc2VecFeb2017MiniCorpus", 40)

# Use demntionality reduction to plot KMeans clusters in 3D
# plotModel3D("Doc2VecFeb2017MiniCorpus", 40)

# Use DBSCAN to cluster. Results were very suboptimal, with most subreddits belonging to a single massive cluster
# newDBSCANModel("Doc2VecFeb2017MiniCorpus", "yourOutput.pkl")



# Feature Forthcoming: Classification
# model = Doc2Vec.load("clusteredModel")
#
# classifier = svm.SVC
# classifier.fit(train_array, train_labels)
# print(classifier.score(test_array, test_labels))










