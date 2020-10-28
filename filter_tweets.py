import csv
import json

from langdetect import detect

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from nltk.tokenize import TweetTokenizer

from collections import defaultdict

from sklearn.metrics import recall_score, precision_score, accuracy_score

tknzr = TweetTokenizer()

def bow(s):
    tkns = tknzr.tokenize(s)
    counts = defaultdict(int)
    for t in tkns:
        t = t.lower()
        counts[t] += 1

    return counts

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

    with open('unfiltered_tweets.txt', 'r') as json_file:
        count = 0
        gender_count = 0
        for line in json_file.readlines():
            try:
                tweet = json.loads(line)
            except:
                continue

            if not tweet['includes']['users']:
                continue

            if not tweet['includes']['users'][0]['name']:
                continue

            name = tweet['includes']['users'][0]['name'].split()[0]
            gender = name_to_gender.get(name)
            if not gender:
                continue

            text = tweet['data']['text']
            try:
                language = detect(text)
            except:
                continue

            if language != 'en':
                continue

            print(gender, text)
            x_data.append(bow(text))
            y_data.append(0 if gender == 'M' else 1)
            gender_count += (0 if gender == 'M' else 1)
            count += 1

        print(count, gender_count / count)

    x_train = x_data[:2500]
    x_test = x_data[2500:]

    y_train = y_data[:2500]
    y_test = y_data[2500:]

    v = DictVectorizer(sparse=False)
    X = v.fit_transform(x_train)

    lgr = LogisticRegression()

    lgr.fit(X, y_train)

    predictions = lgr.predict(v.transform(x_test))
    print(accuracy_score(y_test, predictions), precision_score(y_test, predictions), recall_score(y_test, predictions))

    importance = lgr.coef_
    inverse = v.inverse_transform(importance)[0]

    items = []
    for word in inverse:
        items.append([word, inverse[word]])

    items.sort(key = lambda x: abs(x[1]), reverse=True)
    print(items[:1000])

if __name__ == '__main__':
    main()
