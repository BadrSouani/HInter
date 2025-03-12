import csv

import pandas as pd

import methods


def addDict(d, word, list):
    if d.get(list) is None:
        d[list] = []
    if word not in d[list]:
        d[list].append(word)

all_dicts = 'data/SBIC.v2/SBIC.v2.agg.dev.csv'

file_content = pd.read_csv(all_dicts, delimiter=',').to_numpy()
word_list = file_content[:,2:4]

d = dict()

for row in word_list:
    word, list_type = row[0], row[1]
    if word == '' or word == '[]' or list_type == '' or list_type == '[]': continue
    if '[' not in word and '[' not in list_type: continue    #if valid line

    word = word[1:-1].replace('"','').replace('.','').lower()  #delete " and [ and spaces
    list_type = list_type[1:-1].replace('"','').replace('.','').lower()
    word_array = []
    list_type_array = []

    if ',' in word:
        word_array = word.split(',')
    else :
        word_array.append(word)

    if ',' in list_type:
        list_type_array = list_type.split(',')
    else :
        list_type_array.append(list_type)

    for x in word_array:
        for y in list_type_array:
            addDict(d, x.strip(), y.strip())

#write files
path_list = 'data/gen_list'
keys = list(d.keys())
for key in keys:
    csv_label = path_list + '/' + key + ".csv"
    csv_label_file = methods.createOpen(csv_label, 'w')
    csv_label_writer = csv.writer(csv_label_file, delimiter=';')

    for line in d[key]:
        csv_label_writer.writerow([line])
    csv_label_file.close()

print("Exit")