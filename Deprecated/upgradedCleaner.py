import ftfy
import multiprocessing as mp
import os
import time
import re
import json
import fnmatch

# Based on code from http://www.blopig.com/blog/2016/08/processing-large-files-using-python/ by Alistair Martin
# Warning: as of yet this cleaner can only take 1 GB input files max, though will read through however many you give it
# Recommendation: Use fsplit or similar software.

def jsonCleaner(jsonl, filter, writeList, count):
    """Reads and cleans the JSONL input files."""

    for jsonFile in jsonl:
        try:
            # loads the line as a dictionary
            d = json.loads(jsonFile, encoding="utf-8")
            body = d["body"]
            # deletes URLs
            body = re.sub(r'https?:\/\/.*[\r\n]*', '', body, flags=re.MULTILINE)
            # fixes any unicode encoding errors
            body = ftfy.fix_text(body)
            # deletes chosen punctuation
            body = body.translate(filter)
            body = body.lower()
            # eliminates empty, deleted or removed comments
            if re.search('[a-z]', body) \
                    and body != " removed " \
                    and body != " deleted " \
                    and "your submission has been automatically removed" not in body \
                    and d["author"] != "AutoModerator":
                # this line can be changed to include subreddits or other pieces of data in addition to the comment body
                writeList.append("{}\n".format(body))
                count += 1
            # every 500 000th line it will spit out the 10 most recent comments
            if count == 500000:
                print(writeList[-10:])
                count = 0
        # handles invalid dictionaries
        except ValueError:
            print("Decoding Error")

def textCleaner(text, filter, writeList, count):
    for line in text:
        # deletes URLs
        line = re.sub(r'https?:\/\/.*[\r\n]*', '', line, flags=re.MULTILINE)
        # fixes unicode encoding errors
        line = ftfy.fix_text(line)
        # deletes chosen punctuation
        line = line.translate(filter)
        line = line.lower()
        if re.search('[a-z]', line):
            writeList.append(line + "\n")
            count += 1
        if count == 500000:
            print(writeList[-10:])
            count = 0

def process_wrapper(chunkStart, chunkSize, inputFile):
    # custom translation table to ensure desired punctuation is removed
    punctuation = "–~`!@#$%^&*,;.?:\/]}[{()\"_\\>|<+=\n\r"
    replace = "                                  "
    writeList = []
    count = 0
    transtable = str.maketrans(punctuation, replace)
    with open("inputFiles\\" + inputFile, encoding="utf-8-sig") as f:
        # finds the start of the file chunk
        f.seek(chunkStart)
        # reads the file chunk, spliting it by line
        lines = f.read(chunkSize).splitlines()
        # if inputFile ends in .jsonl, run jsonCleaner
        if fnmatch.fnmatch(inputFile, '*.jsonl'):
            jsonCleaner(lines, transtable, writeList, count)
        # elif inputFile ends in .txt, run textCleaner
        elif fnmatch.fnmatch(inputFile, '*.txt'):
            textCleaner(lines, transtable, writeList, count)
        with open("Feb2017.txt", "a", encoding="utf-8") as w:
            for element in writeList:
                w.write(element)


def chunkify(fname, size=10000000*1024):
    fileEnd = os.path.getsize(fname)
    with open(fname,'r') as f:
        chunkEnd = f.tell()
        while True:
            chunkStart = chunkEnd
            f.seek(size, 0)
            f.readline()
            chunkEnd = f.tell()
            yield chunkStart, chunkEnd - chunkStart
            if chunkEnd > fileEnd:
                break


startTimeClean = time.time()
listOfFiles = os.listdir("inputFiles")

if __name__ == '__main__':
    mp.freeze_support()
    #init objects
    pool = mp.Pool(4)
    jobs = []
    for inputFile in listOfFiles:
        #create jobs

        for chunkStart,chunkSize in chunkify("inputFiles\\" + inputFile):
            jobs.append(pool.apply_async(process_wrapper,(chunkStart,chunkSize, inputFile)) )

        #wait for all jobs to finish
        for job in jobs:
            job.get()

    #clean up
    pool.close()

    #timekeeping
    seconds = (time.time() - startTimeClean)
    minutes = int(seconds/60)
    seconds %= 60
    print("The program cleaned the files in {} minutes and {} seconds.".format(minutes, round(seconds, 2)))
