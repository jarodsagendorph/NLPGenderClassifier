import csv
import json

from langdetect import detect

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from nltk.tokenize import TweetTokenizer
from nltk import pos_tag

from collections import defaultdict

from sklearn.metrics import recall_score, precision_score, accuracy_score

import emoji

tknzr = TweetTokenizer()

def bow(s):
    tkns = tknzr.tokenize(s)
    counts = defaultdict(int)
    for t in tkns:
        t = t.lower()
        counts[t] += 1
        # counts[pos] += 1

    return counts, len(tkns)

def load_names():
    name_to_gender = {}

    with open('names.txt') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            name_to_gender[row['name']] = row['gender']

    return name_to_gender

def main():
    name_to_gender = load_names()

    reader = csv.reader(open('names.txt', 'r'))
    dict = {k: v for k, v in reader}

    x_data = []
    y_data = []

    with open('tweets_today.txt', 'r') as json_file:
        total_count = 0
        count = 0
        gender_count = 0
        word_count = 0
        users = {}
        duplicate_count = 0
        for line in json_file.readlines():
            try:
                tweet = json.loads(line)
            except:
                continue

            total_count += 1

            try:
                if not tweet['includes']['users']:
                    continue

                if not tweet['includes']['users'][0]['name']:
                    continue

                name = tweet['includes']['users'][0]['name'].split()[0]

                gender = name_to_gender.get(name)
                if not gender:
                    continue

                text = tweet['data']['text']
                language = detect(text)

                if language != 'en':
                    continue
            except:
                continue

            print(line)

if __name__ == '__main__':
    main()
