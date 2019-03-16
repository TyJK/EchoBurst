from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import logging
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


class MySentences(object):
    """Takes the corpus file as input, then iterates over it and yields a generator. Must be within the
    __iter__ function as w2v iterates over it twice. Must be a generator to not take up all your RAM.
    Used to create the initial Phraser object"""

    def __init__(self, fileName):
        self.fileName = fileName

    def __iter__(self):
        for line in open(self.fileName, encoding="utf8"):
            yield line.split()


class PhrasingIterable(object):
    """Similar to the first one, but is used to create a version of the corpus generator with phrases.
    You could run a phrasified corpus through this to create trigrams if you wanted"""
    def __init__(self, phrasifier, filesName):
        self.phrasifier, self.fileName = phrasifier, filesName

    def __iter__(self):
        for line in open(self.fileName, encoding="utf8"):
            yield self.phrasifier[line.split()]



def wordMath(additive, negative, model):
    """For word math problems, runs the function and then formats it"""

    result = model.wv.most_similar(positive=additive, negative=negative, topn=3)
    if not negative:
        answer = ("{} = {}".format(' + '.join(additive), result))
    elif not additive:
        answer = ("- {} = {}".format(' - '.join(negative), result))
    else:
        answer = ("{} - {} = {}".format(' + '.join(additive), ' - '.join(negative), result))

    return answer

def extractTests(testFile):
    """Lets you read a file of test results for the word math equations, and reverse engineer them so you can run them
    on a new model"""

    with open(testFile, "r") as r:
        total = []
        for line in r:
            equation, positive, negative = [], [], [],
            switch = 0
            for word in line.split(" "):
                if word == '-':
                    switch = 1
                else:
                    if switch == 0 and (word.isalpha() or "_" in word):
                        positive.append(word)
                    elif word.isalpha():
                        negative.append(word)
                    else:
                        pass
            equation = [positive, negative]
            total.append(equation)
        return total

def testingSuite(modelFile):
    """The rest of the testing suite. Currently hard coded and the auto-tester will only take files originally generated
    from the custom test files (or something in the same format) stored in files called basicWordMath.txt,
    basicSimilarity.txt and basicOddOneOut.txt respectively. Will be modularized and made more flexible in the
    future"""

    reports = []
    model = Word2Vec.load("Models\\" + modelFile)
    choice = input("Press '1' to run custom tests, or '2' to run a predefined set from a file?\n")
    if choice == '1':
        testProcedure = "_customTestResults"
        testChoice = input("Press '1' to do Word Math, '2' to compare similar words, or '3' to find the odd word out.\n")
        if testChoice == '1':
            testType = "_wordMath"
            while True:
                additive = input("Input the words you wish to add, separated by spaces: ").lower().split()
                negative = input("Input the words you wish to subtract, separated by spaces: ").lower().split()
                result = wordMath(additive, negative, model)
                reports.append(result)
                reportChoice = input("Press 'y' to add another equation\n ")
                if reportChoice != 'y':
                    break
        elif testChoice == '2':
            testType = "_similarity"
            while True:
                first = input("Enter the first word to be compared: ").lower()
                second = input("Enter the second word to be compared: ").lower()
                result = model.wv.similarity(first, second)
                reports.append("The similarity between {} and {} is {}.".format(first, second, result))
                reportChoice = input("Press 'y' to add another equation\n ")
                if reportChoice != 'y':
                    break
        elif testChoice == '3':
            testType = "_oddOneOut"
            while True:
                comparison = input("Enter the list of words to be compared, separated by a space: ").lower()
                result = model.wv.doesnt_match(comparison.split())
                reports.append("{} : {} is the odd one out".format(comparison, result))
                reportChoice = input("Press 'y' to add another equation\n ")
                if reportChoice != 'y':
                    break
        else:
            print("Invalid Input")
    elif choice == '2':
        testProcedure = "_automaticTestResults"
        testChoice = input("Press '1' to do Word Math, '2' to compare similar words, or '3' to find the odd word out.\n")
        if testChoice == '1':
            testType = "_wordMath"
            tests = extractTests("modelTests\\basicWordMath.txt")
            for test in tests:
                additive = test[0]
                negative = test[1]
                reports.append(wordMath(additive, negative, model))
        elif testChoice == '2':
            testType = "_similarity"
            tests = open("modelTests\\basicSimilarity.txt", "r").readlines()
            for test in tests:
                if len(test) > 1:
                    word = test.split(" ")
                    first = word[3]
                    second = word[5]
                    result = model.wv.similarity(first, second)
                    reports.append("The similarity between {} and {} is {}.".format(first, second, result))
        elif testChoice == '3':
            testType = "_oddOneOut"
            lines = open("modelTests\\basicOddOneOut.txt", "r").readlines()
            for line in lines:
                words = line.split()
                testWords = []
                for word in words:
                    if word == ":":
                        break
                    else:
                        testWords.append(word)
                result = model.wv.doesnt_match(testWords)
                reports.append("{} : {} is the odd one out".format(" ".join(testWords), result))
        else:
            print("Invalid Input")
    else:
        print("Invalid input")

    with open("modelTests\\" + modelFile + testProcedure + testType + ".txt", "a", encoding="utf8") as f:
        for report in reports:
            f.write(report + "\n")
            print(report)


#Track results
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#The output file name
modelFile = "Feb2017FullCorpus300D"

#Original corpus available at: http://files.pushshift.io/reddit/comments/ (RC_2017-02.BZ2)
#Retreieves the corpus file. This corpus was generated using the output from the upgradedCleaner.py file
sentences = MySentences("Data\\Feb2017.txt")

#Creates a Phrases object from the corpus
myPhrases = Phrases(sentences, min_count=20)

#Creates a much smaller Phraser object from the Phrases object
bigram_transformer = Phraser(myPhrases)

#Saves it so you don't have to redo this every time.
bigram_transformer.save("Feb2017BigramTransformer")
bigram_transformer = Phraser.load("Feb2017BigramTransformer")

#Create and save the actual model
model = Word2Vec(PhrasingIterable(bigram_transformer, "Data\\Feb2017.txt"), min_count=15, workers=4, size=300, window=8)
model.save('Models\\' + modelFile)
model = Word2Vec.load('Models\\' + modelFile)


# Accuracy tests
model.accuracy('questions-words.txt')
testingSuite(modelFile)
