import os
import pandas as pd
from collections import Counter
if __name__ == '__main__':
    names = dict()
    directory = r'./names'
    for filename in os.scandir(directory):
        if filename.path.endswith(".txt"):
            print(filename)
            file = pd.read_csv(filename, header=None)
            for index, row in file.iterrows():
                if row[0] not in names:
                    names[row[0]] = {'M': 0, 'F': 0}
                names[row[0]][row[1]] += row[2]
    datafile = open("names.txt", "w")
    for item in names.items():
        occ = item[1]["M"] + item[1]["F"]
        m_prob = float(item[1]["M"])/float(occ)
        if m_prob >= 0.99 and occ > 1000:
            string = item[0].lower() + ",M\n"
            datafile.write(string)
        if m_prob <= 0.01 and occ > 1000:
            string = item[0].lower() + ",F\n"
            datafile.write(string)
    datafile.close()
