from gensim.models import Doc2Vec

model = Doc2Vec.load("Models\\Doc2VecFeb2017FullCorpus")

while True:
    choice = input("Press 1 to compare documents within the model to each other.\n"
                   "Press 2 to run similarity tests on individual words.\n"
                   "Press 3 to get the top related subreddits for an inferred new vector (comment).\n"
                   "Hit any key to exit.\n")
    # Allows you to find related subreddits. Just a sanity check. Works decently well in the current state
    if choice == "1":
        docChoice = input("Enter the subreddit you want to test.\n")
        print(model.docvecs.most_similar(docChoice))
    # Similar to the word2vec .most_similar() method.
    elif choice == "2":
        wordChoice = input("Enter the word you wish to analyze.\n").lower()
        print(model.most_similar(wordChoice))
    # Feed this the 'testing.txt' file which contains distantly labelled comments that were found by searching for key
    # strings within the comments. Better testing data will come along with scraped data in the coming weeks
    elif choice == "3":
        with open("testing.txt") as t:
            resultList = []
            testDocs = t.readlines()
            for doc in testDocs:
                doc = doc.split(",")
                tag = doc[0]
                body = doc[1]
                newVec = model.infer_vector(body.split())
                resultList.append("The origial category is {}: {} {}\n".
                                  format(tag, body, model.docvecs.most_similar(positive=[newVec])))
            with open("results.txt", "a") as t:
                for element in resultList:
                    t.write(element)
    else:
        break
