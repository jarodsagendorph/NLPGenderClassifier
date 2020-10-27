import csv
import json

def main():
    reader = csv.reader(open('names.txt', 'r'))
    dict = {k: v for k, v in reader}

    with open('unfiltered_tweets.txt', 'r') as json_file:
        for line in json_file.readlines():
            print(line)
            json.loads(line)

if __name__ == '__main__':
    main()
