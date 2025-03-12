import argparse
import os

import torch
import datasets
import time
from tools import methods
from tools import utils as sm

parser = argparse.ArgumentParser(description='Process values.')
parser.add_argument('dataset_name', type=str, help="Name of the dataset to use")
parser.add_argument('dict_path', type=str, help="Path to words file")
parser.add_argument('set', type=str, help="Set to train on between train, validation and test")
parser.add_argument('--inter_dict_path', type=str, default="",
                    help="Path to the second words file for intersectional method")
parser.add_argument('--output_path', type=str, default="../output/", help="Path to the output folder")
parser.add_argument('--data_path', type=str, default="../data/", help="Path to the data folder")

args = parser.parse_args()
dataset_name = args.dataset_name
dict_path = args.dict_path
inter_dict_path = args.inter_dict_path
split_of_dataset = args.set
output_folder_path = args.output_path
data_path = args.data_path


word_file_name, new_word_list_A, new_word_list_B = methods.getWordlist(dict_path)
upper1 = True if "race" in dict_path else False
if inter_dict_path != "":
    inter_file_name, inter_list_A, inter_list_B = methods.getWordlist(inter_dict_path)
    full_lists_name = word_file_name + '+' + inter_file_name
    is_intersectional = True
    upper2 = True if "race" in inter_dict_path else False
else:
    is_intersectional = False
    full_lists_name = word_file_name


output_path = output_folder_path + "/mutants/"+\
              "_".join(["llm", split_of_dataset, dataset_name, split_of_dataset, word_file_name +
                        ("" if not is_intersectional else "+"+inter_file_name) +'.pkl'])
previous_discarded = 0
previous_time = 0
previous_mutants = []
previous_occurrences = 0

modified = True
if os.path.isfile(output_path):
    modified = False
    print("File exist - Resuming Generation.")
    content = methods.getFromPickle(output_path, "rb")
    previous_discarded = content[0]['discarded_test_cases_sentence']
    previous_occurrences = content[0]['occurrences']
    previous_time = content[0]['time (s)']
    previous_mutants = content[1]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
isMulti = True if dataset_name in ["ecthr_a", "ecthr_b", "eurlex"] else False

# For each case, join the sentences in order to represent the case into one string
if dataset_name == 'imdb':
    dataset_path = data_path+"imdb/imdb.csv"
    dataset_text = methods.getFromCSV(dataset_path, delimiter=",", header=0)[:, 0]
    complete_dataset_size = len(dataset_text)
    split_of_dataset = "None"

else :
    complete_dataset = datasets.load_dataset("lex_glue", dataset_name, split=split_of_dataset)
    complete_dataset_size = len(complete_dataset)
    dataset_text = [''.join(complete_dataset[i][('text')]) for i in
                    range(complete_dataset_size)]
dataset_text = methods.format_dataset(dataset_text)


print(f"Using dataset named {dataset_name} {split_of_dataset} set of length {complete_dataset_size}.\n"
      f"Word file used is {full_lists_name}.")

start_time = time.time()

def fct_modification(fct_replacement, word_list, replacement_list=[]):
    discarded_mutants_length = 0
    dependencies = [None] * complete_dataset_size
    original_sentences_array = [None] * complete_dataset_size
    l_replaced_occurrences = 0
    mutants = []
    for i_word in range(len(word_list)):
        print("- ", i_word+1, "/", len(word_list))
        word = word_list[i_word]
        replacement_word = fct_replacement(i_word, replacement_list)
        for i in range(complete_dataset_size):
            modified_text, occurrences = methods.replace(dataset_text[i], word, replacement_word, upper1)
            if occurrences > 0:
                l_replaced_occurrences += occurrences

                if original_sentences_array[i] == None:
                    original_sentences_array[i] = sm.split_into_sentences(dataset_text[i])
                mutant_sentences = sm.split_into_sentences(modified_text)
                if dependencies[i] == None:
                    dependencies[i] = [None] * len(original_sentences_array[i])
                similarity_info= sm.isTextStructureSimilar(original_sentences_array[i], dependencies[i], mutant_sentences)
                similar = similarity_info[0]
                discarded_mutants_length += similarity_info[4]

                mutants.append([word, replacement_word, modified_text, i, dataset_text[i], occurrences, similar])

    return l_replaced_occurrences, mutants , discarded_mutants_length


def fct_intersectional(word_list_1, replacement_list_1, word_list_2, replacement_list_2, modified):
    discarded_mutants_length = previous_discarded
    dependencies = [None]*complete_dataset_size
    original_sentences_array = [None]*complete_dataset_size
    mutants = previous_mutants
    absence = set()
    start = 0
    new_content = False
    if len(mutants) != 0 :
        start = 1 + mutants[-1][5]
        print('Starting from ', start)
    for i in range(start, complete_dataset_size):
        if (i+1)%100 == 0:
            print("---", i + 1, "/", complete_dataset_size)
            if new_content:
                print("Checkpoint saved.")
                generation_time = time.time() - start_time + previous_time

                if is_intersectional:
                    header = ['Word 1', 'Replacement 1', 'Word 2', 'Replacement 2', 'Mutant', 'Index', 'Original',
                              "Similarity"]
                else:
                    header = ['Word', 'Replacement', 'Mutant', 'Index', 'Original', "Occurrences", "Similarity"]

                dict_results = {"time (s)": generation_time, "occurrences": occurrences,
                                "header": header, 'discarded_test_cases_sentence': discarded_mutants_length}
                lines = [dict_results, mutants]
                methods.writePickle(lines, output_path, "wb")
                new_content = False
        for i_word in range(len(word_list_1)):
            word_1 = word_list_1[i_word]
            replacement_word_1 = replacement_list_1[i_word]
            key1 = tuple([i, word_1])
            if key1 in absence: continue
            modified_text_w1, occurrences = methods.replace(dataset_text[i], word_1, replacement_word_1, upper1)
            if occurrences == 0:
                absence.add(key1)
                continue
            for y_word in range(len(word_list_2)):
                word_2 = word_list_2[y_word]
                key2 = tuple([i, word_2])
                if key2 in absence: continue
                if word_1 == word_2 or word_2 == replacement_word_1: continue
                replacement_word_2 = replacement_list_2[y_word]
                modified_text_w1_w2, occurrences2 = methods.replace(modified_text_w1, word_2, replacement_word_2, upper2)
                if occurrences2 == 0:
                    absence.add(key2)
                    continue

                if original_sentences_array[i] == None:
                    original_sentences_array[i] = sm.split_into_sentences(dataset_text[i])
                mutant_sentences = sm.split_into_sentences(modified_text_w1_w2)
                if dependencies[i] == None:
                    dependencies[i] = [None] * len(original_sentences_array[i])
                similarity_info = sm.isTextStructureSimilar(original_sentences_array[i], dependencies[i], mutant_sentences)
                similar = similarity_info[0]
                discarded_mutants_length += similarity_info[4]

                mutants.append([word_1, replacement_word_1, word_2, replacement_word_2, modified_text_w1_w2, i, dataset_text[i], similar])
                modified=True
                new_content = True
    if modified == False :
        print("No modification made - exiting")
        exit(0)
    return 0, mutants, discarded_mutants_length


def fct_replacement(list_word, list_replacement):
    def replacement(it, list_of_word): return list_of_word[it]

    return fct_modification(replacement, list_word, list_replacement)




if is_intersectional: occurrences, mutants, discarded = fct_intersectional(new_word_list_A, new_word_list_B, inter_list_A, inter_list_B, modified)
else:occurrences, mutants, discarded = fct_replacement(new_word_list_A, new_word_list_B)

generation_time = time.time() - start_time + previous_time

if is_intersectional : header = ['Word 1', 'Replacement 1', 'Word 2', 'Replacement 2', 'Mutant', 'Index', 'Original', "Similarity"]
else : header = ['Word', 'Replacement', 'Mutant', 'Index', 'Original', "Occurrences", "Similarity"]

dict_results = {"time (s)" : generation_time, "occurrences" : occurrences,
                "header" : header, 'discarded_test_cases_sentence':discarded}
lines = [dict_results, mutants]

methods.writePickle(lines, output_path, "wb")

print(f"Done generating {len(mutants)} mutants for {generation_time} seconds.")
