import csv
import json

import random

from langdetect import detect

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from nltk.tokenize import TweetTokenizer
import nltk

from collections import defaultdict

from sklearn.metrics import recall_score, precision_score, accuracy_score

# import torch
# from transformers import BertTokenizer, BertModel

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# model = BertModel.from_pretrained('bert-base-uncased',
#   output_hidden_states = True, # Whether the model returns all hidden-states.
# )

import os

def read_files(folder):
    books = {}
    for filename in os.listdir(os.path.join(os.getcwd(), folder)):
        with open(os.path.join(os.path.join(os.getcwd(), folder), filename), 'r') as f:
            author = filename.split("||")[0]
            if author not in books:
                books[author] = []
            books[author].append(f.read())

    return books

tknzr = TweetTokenizer()

def encode(s):
  text = s
  marked_text = "[CLS] " + text + " [SEP]"
  tokenized_text = tokenizer.tokenize(marked_text)
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

  segments_ids = [1] * len(tokenized_text)

  tokens_tensor = torch.tensor([indexed_tokens])
  segments_tensors = torch.tensor([segments_ids])

  with torch.no_grad():

    outputs = model(tokens_tensor, segments_tensors)

    # Evaluating the model will return a different number of objects based on 
    # how it's  configured in the `from_pretrained` call earlier. In this case, 
    # becase we set `output_hidden_states = True`, the third item will be the 
    # hidden states from all layers. See the documentation for more details:
    # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
    hidden_states = outputs[2]
  
  token_vecs = hidden_states[-2][0]
  temp = torch.mean(token_vecs, dim=0)
  return temp.numpy()

def bow(s):
    tkns = tknzr.tokenize(s)
    counts = defaultdict(int)
    last_pos = None
    last_word = None
    for tup in nltk.pos_tag(tkns):
        t,pos = tup
        if t.startswith('@'):
            continue

        if t.startswith('http'):
            continue

        if last_pos:
            counts["POS " + last_pos + " " + pos] += 1
            # counts[last_word + " " + t] += 1
        counts["POS " + pos] += 1

        last_pos = pos
        last_word = t

        # t = t.lower()
        # counts[t] += 1

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

    x_data = []
    y_data = []

    tw = {
        'M': [],
        'F': []
    }

    with open('no_rt.txt', 'r') as json_file:
        count = 0
        gender_count = 0
        for line in json_file.readlines():
            try:
                tweet = json.loads(line)
            except:
                continue

            name = tweet['includes']['users'][0]['name'].split()[0]
            gender = name_to_gender.get(name)
            if not gender:
                continue

            text = tweet['data']['text']
            # print(len(text))
            # if len(text) < 35:
            #     continue

            parts = [p for p in text.split() if p.lower() in ["i", "i'm", "i'll", "i'd", "me", "mine", "we", "our"]]
            # text_len = " ".join(parts)

            if not parts:
                continue

            print(text)
            tw[gender].append(bow(text))

            # print(gender, text)
            # x_data.append(bow(text))
            # y_data.append(0 if gender == 'M' else 1)
            gender_count += (0 if gender == 'M' else 1)
            count += 1

        print(count, gender_count / count)

    print(len(tw['M']), len(tw['F']))

    train_size = 800
    test_size = 200

    predictions = []
    actuals = []

    print("here")

    for i in range(1):
        random.shuffle(tw['M'])
        random.shuffle(tw['F'])

        x_train = tw['M'][:train_size] + tw['F'][:train_size]
        x_test = tw['M'][train_size:(train_size+test_size)] + tw['F'][train_size:(train_size+test_size)]

        y_train = [0] * train_size + [1] * train_size
        y_test = [0] * test_size + [1] * test_size

        v = DictVectorizer(sparse=False)
        x_train = v.fit_transform(x_train)
        x_test = v.transform(x_test)

        lgr = LogisticRegression(max_iter=1000)

        lgr.fit(x_train, y_train)

        predictions += list(lgr.predict(x_test))
        actuals += y_test
    print(accuracy_score(actuals, predictions), precision_score(actuals, predictions), recall_score(actuals, predictions))

    importance = lgr.coef_
    inverse = v.inverse_transform(importance)[0]

    items = []
    for word in inverse:
        items.append([word, inverse[word]])

    items.sort(key = lambda x: abs(x[1]), reverse=True)
    print(items[:100])

if __name__ == '__main__':
    main()
