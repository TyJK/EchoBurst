from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from logging import basicConfig, INFO

class LabeledLineSentence(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename, encoding="utf-8"):
            csv = line.split(",")
            subreddit = csv[0]
            body = csv[1].split()
            yield TaggedDocument(words=body, tags=[subreddit])



basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=INFO)

# Original corpus available at: http://files.pushshift.io/reddit/comments/ (RC_2017-02.BZ2)
# Retreieves the corpus file. This corpus was generated using the output from the upgradedCleaner.py file
documents = LabeledLineSentence("Data\\Feb2017SubredditIncluded.txt")

outputFile = "Doc2VecFeb2017FullCorpus"

# These are the parameters we've used so far, using the full corpus to build our model
# It was, in the end, over 1.5 gb, so can not be directly uploaded to GitHub
model = Doc2Vec(dm=0, alpha=0.025, min_alpha=0.025, dm_mean=1, size=300, window=5, negative=5, min_count=10, workers=12)
model.build_vocab(documents)

for epoch in range(5):
    model.train(documents, total_examples=model.corpus_count)
    # decrease the learning rate
    model.alpha -= 0.002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save(outputFile)