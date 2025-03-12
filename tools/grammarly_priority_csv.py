import re

import datasets
import random
from tools import methods

def replace(case, w_1, r_1, w_2=None, r_2=None):
    regex1 = r"\b" + w_1 + r"\b"
    insensitive_word1 = re.compile(regex1, re.IGNORECASE)
    temp_text = insensitive_word1.sub(r_1, case)
    if w_2 == None or r_2 == None: return temp_text
    regex2 = r"\b" + w_2 + r"\b"
    insensitive_word2 = re.compile(regex2, re.IGNORECASE)
    return insensitive_word2.sub(r_2, temp_text)

N=10000
gender_list = ['female_male', 'female_male_job', 'male_female', 'male_female_job']
race_list = ['african_american', 'african_arab', 'african_asian', 'african_european', 'american_african', 'american_arab', 'american_asian',
             'american_european','arab_african','arab_american','arab_asian','arab_european','asian_african','asian_american',
             'asian_arab','asian_european','european_african','european_american','european_arab','european_asian','majority_minority',
             'majority_mixed','minority_majority','minority_mixed','mixed_majority','mixed_minority']
body_list = ['common_disorder', 'common_hair', 'common_uncommon', 'disorder_common', 'old_young', 'uncommon_common', 'young_old', 'hair_common']
models = ['bert-base-uncased', 'microsoft/deberta-base', 'roberta-base', 'nlpaueb/legal-bert-base-uncased']
sets = ["validation", "test", "train"]
dts= ['scotus', 'ecthr_a', 'ecthr_b', 'ledgar', 'eurlex']
# dts= ['scotus']
dict_datasets = dict()
dict_mutants = dict()
dict_errors = dict()
print(">Stocking Originals, Mutants, and Errors ...")
for d in dts:
    for s in sets:
        complete_dataset = datasets.load_dataset('lex_glue', d, split=s)
        complete_dataset_size = len(complete_dataset)
        dict_datasets[d+s] = [''.join(complete_dataset[i]['text']).replace(";", '.').replace("\n", '.') for i in
                    range(complete_dataset_size)]
        for values in [['replacement_replacement_atomic', [body_list, race_list, gender_list], [0, 1, 3], [0, 1, 4]],
                       ['intersectional_intersectionality',
                        [[i+'+'+j for i in body_list for j in race_list],
                         [i+'+'+j for i in body_list for j in gender_list],
                          [i+'+'+j for i in race_list for j in gender_list]],
                        [0, 1, 2, 3, 5], [0, 1, 2, 3, 6]]]:
            for lists in values[1]:
                for l in lists:
                    for m in models:
                        error_content = methods.getFromPickle(
                            '/'.join(['..', 'output', d, m, s, 'error_details', values[0]+'-'+l+'.pkl']), 'rb')
                        dict_errors[d + m + s + l] = set( [tuple([x[y] for y in values[3]]) for x in error_content if x[-1]==0] )#CHANGE THIS TO ONLY HAVE SIMILAR OR NOT SIMILAR
                    mutant_content = methods.getFromPickle('../output/mutants/' + '_'.join([d, s, l + '.pkl']), 'rb')
                    mutants = []
                    for mutant in mutant_content:
                        if mutant[-1] == 0 : continue
                        if len(dict_datasets[d+s][mutant[values[2][-1]]])> 32680:
                            continue
                        nb_errors = 0
                        keys = tuple([mutant[y] for y in values[2]])
                        for m in models:
                            if keys in dict_errors[d + m + s + l] : nb_errors +=1
                        mutants.append(keys+tuple([nb_errors]))
                    mutants.sort(key = lambda a:a[-1], reverse=True)
                    dict_mutants[d + s + l] = mutants

print("Selecting cases ...")
originals_used = set()
rows_original = [['Case', 'Dataset', 'Set', 'Case Number']]
for d in dts:
    print("Doing",d)
    for values in [[[body_list, race_list, gender_list], [2, 3, 0, 1], [0, 1, None, None], "atomic"],
                   [[[i + '+' + j for i in body_list for j in race_list],
                    [i + '+' + j for i in body_list for j in gender_list],
                    [i + '+' + j for i in race_list for j in gender_list]],
                    [4, 5, 0, 1, 2, 3], [0, 1, 2, 3], "intersectional"]]:
        rows = [['Case', 'Dataset', 'Set', 'Case Number', 'Nb_errors', 'w1', 'r1', 'w2', 'r2']]
        cases_taken = []
        for lists in values[0]:
            cases_taken_error = []
            cases_taken_nerror = []
            for s in sets:
                for l in lists:
                    for c in dict_mutants[d + s + l]:
                        if len(cases_taken_error)+len(cases_taken_nerror)+len(cases_taken) == N//len(values[0])-(values[0].index(lists)) \
                                and len(cases_taken_error) == N // len(values[0]) // 4:
                            break
                        mutated_case = replace(dict_datasets[d+s][c[values[1][0]]],
                                               c[values[2][0]],
                                               c[values[2][1]],
                                               c[values[2][2]] if values[2][2] != None else None,
                                               c[values[2][3]] if values[2][3] != None else None)
                        if len(mutated_case) > 32680 : continue
                        if c[-1] != 0:
                            if len(cases_taken_error) < N // len(values[0]) // 4:
                                cases_taken_error.append([mutated_case]+[d, s]+[c[y] for y in values[1]])
                                if len(cases_taken_nerror) + len(cases_taken_error) > N // len(values[0]):
                                    cases_taken_nerror.pop(random.randrange(0,len(cases_taken_nerror)))
                        elif c[-1] == 0 :
                            if len(cases_taken_error)+len(cases_taken_nerror) == N // len(values[0]) :
                                cases_taken_nerror.pop(random.randrange(0,len(cases_taken_nerror)))
                            cases_taken_nerror.append([mutated_case]+[d, s]+[c[y] for y in values[1]])
            cases_taken += cases_taken_error
            cases_taken += cases_taken_nerror
        originals_used.update(set([(x[1], x[2], x[3]) for x in cases_taken]))
        rows += cases_taken
        mutants_file = "../output/" + d + "_"+values[3]+"_grammarly.csv"
        methods.writeCSV(mutants_file, rows)
original_file = "../output/Originals_grammarly.csv"
rows_original += [[[dict_datasets[x[0] + x[1]][x[2]]] + list(x)][0] for x in originals_used]
methods.writeCSV(original_file, rows_original)
print("Done")