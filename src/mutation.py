import argparse
import numpy as np
import torch
import transformers
import datasets
import re
import cuda
import os
import csv
import pickle
import time
from tools import utils as sm
from tools import methods

parser = argparse.ArgumentParser(description='Process values.')
parser.add_argument('model', type=str, help="Path to the model")
parser.add_argument('dataset_path', type=str, help="Hugging face path to dataset")
parser.add_argument('dict_path', type=str, help="Path to words file")
parser.add_argument('description', type=str, help="Technique description")
parser.add_argument('method', type=str,
                    help="Method to use for text modification (replacement, deletion, intersectional)")
parser.add_argument('set', type=str, help="Set to train on between train, validation and test")
parser.add_argument('--inter_dict_path', type=str, default="",
                    help="Path to the second words file for intersectional method")
parser.add_argument('--length', type=int, default=512, help="The length truncation of the labels used")
parser.add_argument('--mutation_only', action='store_true', help="Only generate mutants without testing")

args = parser.parse_args()
model_path = args.model
dataset_path = args.dataset_path
dict_path = args.dict_path
inter_dict_path = args.inter_dict_path
trunc_length = args.length
technique = args.description
method_name = args.method
split_of_dataset = args.set
mutation_only = args.mutation_only

if method_name not in ['replacement', 'deletion', 'intersectional']:
    raise Exception('Method for text modification unknown.')

word_file_name, new_word_list_A, new_word_list_B = methods.getWordlist(dict_path)
if method_name == 'intersectional':
    if inter_dict_path == "":
        raise Exception('Intersectional method requires inter_dict_path parameter.')
    inter_file_name, inter_list_A, inter_list_B = methods.getWordlist(inter_dict_path)
    full_lists_name = word_file_name + '+' + inter_file_name
    is_intersectional = True
else:
    is_intersectional = False
    full_lists_name = word_file_name

device = torch.device('cuda') if cuda else torch.device('cpu')
if mutation_only : device = torch.device('cpu')

model_config = transformers.PretrainedConfig.from_pretrained(model_path)

model_prediction_type = model_config.problem_type if model_config.problem_type == 'single_label_classification' else "multi_label_classification"
isMulti = model_config.problem_type != 'single_label_classification'

dict_labels = model_config.id2label
num_labels = len(dict_labels)
model_labels = list(dict_labels.values())
model_name = model_config.name_or_path

# Processing Dataset
dataset_name = model_config.finetuning_task
complete_dataset = datasets.load_dataset(dataset_path, name=dataset_name, data_dir='data', split=split_of_dataset)
complete_dataset_size = len(complete_dataset)

# For each case, join the sentences in order to represent the case into one string
dataset_target_column = 'labels' if isMulti else 'label'
dataset_text = [''.join(complete_dataset[i][('text')]).lower() for i in
                range(complete_dataset_size)]  # IN LOWER CASE
dataset_labels = [complete_dataset[i][dataset_target_column] for i in range(complete_dataset_size)]
if not isinstance(dataset_labels[0], list):
    for x in range(complete_dataset_size): dataset_labels[x] = [dataset_labels[x]]
for x in range(complete_dataset_size): dataset_labels[x].sort()
output_path = "output" + "/" + dataset_name + "/" + model_name + "/" + split_of_dataset + "/"

# Loading Tokenizer and Model
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_path)
model = transformers.AutoModelForSequenceClassification.from_pretrained(
    model_path,
    problem_type=model_prediction_type)
model.to(device)

# truncation of text to replace only used tokens
truncated_text = methods.getTruncated(output_path, dataset_text, tokenizer, trunc_length)

if not mutation_only : model_base_prediction = methods.getPrediction(output_path, truncated_text, tokenizer, model, isMulti, trunc_length,
                                              device)
print("Using model from {}\n defined as {} which name is {}\n with dataset of {} named {}"
      "\n Using {} set of length {}.\nDoing method \'{}\'.\nModel is of type {} and has {} output Labels.\nWord file used is {} with {} tuples."
      .format(model_path, type(model).__name__, model_name, dataset_path, dataset_name,
              split_of_dataset, complete_dataset_size, method_name, model_prediction_type, num_labels, full_lists_name,
              str(len(new_word_list_A)) + ('+' + str(len(inter_list_A)) if is_intersectional else "")))

start_time = time.time()


def is_Different(array1, array2, more_label, less_label):
    base = set(array1)
    prediction = set(array2)
    if not base == prediction:
        difference = base.symmetric_difference(prediction)
        for labels in difference:
            if labels in prediction:
                more_label[labels] += 1
            else:
                less_label[labels] += 1
        return True
    return False


def fct_modification(fct_replacement, word_list, replacement_list=[]):
    l_replaced_occurrences = 0
    total_error = 0
    l_changed_cases = 0
    l_changed_prediction_details = []

    l_more_label_counter = [0] * num_labels
    l_less_label_counter = [0] * num_labels
    mutants = []

    dependencies = [None] * complete_dataset_size
    original_sentences_array = [None] * complete_dataset_size
    discarded = 0
    for i_word in range(len(word_list)):
        total_word_occurrence = 0
        word_error = 0
        word = word_list[i_word]
        replacement_word = fct_replacement(i_word, replacement_list)
        regex = r"\b" + word + r"\b"
        insensitive_word = re.compile(regex)
        for i in range(complete_dataset_size):
            occurrences = len(re.findall(regex, truncated_text[i]))
            if occurrences > 0:
                modified_text = insensitive_word.sub(replacement_word, truncated_text[
                    i])  # Replace all occurences in the dataset, case sensitive
                # Testing Model

                if original_sentences_array[i] == None:
                    original_sentences_array[i] = sm.split_into_sentences(dataset_text[i])
                mutant_sentences = sm.split_into_sentences(modified_text)
                if dependencies[i] == None:
                    dependencies[i] = [None] * len(original_sentences_array[i])
                similarity_info = sm.isTextStructureSimilar(original_sentences_array[i], dependencies[i],
                                                            mutant_sentences)
                similar = similarity_info[0]
                if not similar :discarded+=1
                total_word_occurrence += occurrences
                l_changed_cases += 1
                mutants.append([word, replacement_word, modified_text, i, truncated_text[i], similar])

                if mutation_only or not similar: continue
                predicted_class_id = methods.predict(modified_text, tokenizer, model, isMulti, trunc_length, device)

                # if prediction changed after replacement
                if is_Different(model_base_prediction[i], predicted_class_id, l_more_label_counter,
                                l_less_label_counter):
                    word_error += 1
                    l_changed_prediction_details.append(
                        [word, replacement_word, model_base_prediction[i], predicted_class_id, i, truncated_text[i]])


        l_replaced_occurrences += total_word_occurrence
        total_error += word_error
        if word_error > 0 and not mutation_only:
            print("> Found {} occurrences of \'{}\' for {} error ({} error total, {} elements discarded)."
                  .format(total_word_occurrence, word, word_error, total_error, discarded))
    return l_replaced_occurrences, l_changed_cases, l_changed_prediction_details, l_more_label_counter, l_less_label_counter, mutants, discarded


def fct_intersectional(word_list_1, replacement_list_1, word_list_2, replacement_list_2):
    l_changed_cases = 0
    total_error = 0
    l_changed_prediction_details = []

    l_more_label_counter = [0] * num_labels
    l_less_label_counter = [0] * num_labels

    dependencies = [None]*complete_dataset_size
    original_sentences_array = [None]*complete_dataset_size

    mutants = []
    discarded = 0
    for i in range(complete_dataset_size):
        print("Doing ", i + 1, "/", complete_dataset_size)
        case_error = 0
        case_mutant = 0
        for i_word in range(len(word_list_1)):
            word_1 = word_list_1[i_word]
            replacement_word_1 = replacement_list_1[i_word]
            regex_1 = r"\b" + word_1 + r"\b"
            insensitive_word_1 = re.compile(regex_1)
            if len(re.findall(regex_1, truncated_text[i])) == 0: continue
            modified_text_wi = insensitive_word_1.sub(replacement_word_1, truncated_text[i])

            for y_word in range(len(word_list_2)):
                word_2 = word_list_2[y_word]
                regex_2 = r"\b" + word_2 + r"\b"
                if word_1 == word_2 or word_2 == replacement_word_1: continue
                if len(re.findall(regex_2, modified_text_wi)) == 0: continue
                case_mutant += 1
                replacement_word_2 = replacement_list_2[y_word]
                insensitive_word_2 = re.compile(regex_2)
                modified_text_w1_w2 = insensitive_word_2.sub(replacement_word_2, modified_text_wi)

                if original_sentences_array[i] == None:
                    original_sentences_array[i] = sm.split_into_sentences(dataset_text[i])
                mutant_sentences = sm.split_into_sentences(modified_text_w1_w2)
                if dependencies[i] == None:
                    dependencies[i] = [None] * len(original_sentences_array[i])
                similarity_info = sm.isTextStructureSimilar(original_sentences_array[i], dependencies[i], mutant_sentences)
                similar = similarity_info[0]

                if not similar : discarded+=1

                mutants.append([word_1, replacement_word_1, word_2, replacement_word_2, modified_text_w1_w2, i, truncated_text[i], similar])
                if mutation_only or not similar: continue

                # testing prediction of mutant
                predicted_class_id = methods.predict(modified_text_w1_w2, tokenizer, model, isMulti, trunc_length,
                                                     device)
                if is_Different(model_base_prediction[i], predicted_class_id, l_more_label_counter,
                                l_less_label_counter):
                    case_error += 1
                    l_changed_prediction_details.append(
                        [word_1, replacement_word_1, word_2, replacement_word_2, model_base_prediction[i],
                         predicted_class_id, i, truncated_text[i]])
        l_changed_cases += case_mutant
        if case_error > 0 and not mutation_only:
            total_error += case_error
            print(">>Case {}/{} modified {} time(s) for {} error(s) ({} discarded).".format(i, complete_dataset_size, case_mutant,
                                                                           case_error, discarded))
    print("Total of {} mutants and {} errors.".format(l_changed_cases, total_error))
    return 0, l_changed_cases, l_changed_prediction_details, l_more_label_counter, l_less_label_counter, mutants, discarded


def fct_deletion(list_word):
    unique_word_list = np.array(list(set(list_word)))

    def replacement_deletion(it, l): return ' '

    return fct_modification(replacement_deletion, unique_word_list)


def fct_replacement(list_word, list_replacement):
    def replacement(it, list_of_word): return list_of_word[it]

    return fct_modification(replacement, list_word, list_replacement)


if method_name == 'deletion':
    replaced_occurrences, changed_cases, changed_prediction_details, more_label_counter, \
    less_label_counter = fct_deletion(new_word_list_A)

if method_name == 'replacement':
    replaced_occurrences, changed_cases, changed_prediction_details, more_label_counter, \
    less_label_counter, mutants, discarded = fct_replacement(new_word_list_A, new_word_list_B)



if is_intersectional:
    replaced_occurrences, changed_cases, changed_prediction_details, more_label_counter, \
    less_label_counter, mutants, discarded = fct_intersectional(new_word_list_A, new_word_list_B, inter_list_A, inter_list_B)

testing_time = time.time() - start_time
if mutation_only :
    if method_name == "replacement": mutations_path = "_".join(["output/mutants/"+dataset_name, split_of_dataset, word_file_name+'.pkl'])
    else : mutations_path = "_".join(["output/mutants/"+dataset_name, split_of_dataset, word_file_name+"+"+inter_file_name+'.pkl'])
    methods.writePickle(mutants, mutations_path, "wb")
    csv_file = (output_path+"/generation_time/" + "_".join([dataset_name, split_of_dataset, word_file_name+'.pkl'])) if method_name == "replacement" else \
        output_path +"/generation_time/" + "_".join([dataset_name, split_of_dataset, word_file_name+"+"+inter_file_name+'.pkl'])
    methods.writePickle([["time (s)", testing_time]], csv_file, "wb")
    print("Done")
    exit(0)


print("Testing took {} seconds.".format(testing_time))

# model_name = "bert-base-uncased"
num_error = len(changed_prediction_details)
err_rate = num_error / changed_cases if changed_cases != 0 else 'None'

# result file oppening
csv_file = output_path + "result" + ".csv"
csv_exist = os.path.isfile(csv_file)
print("Saving results in {}.".format(csv_file))
csvfile = methods.createOpen(csv_file, 'a')
csv_writer = csv.writer(csvfile, delimiter=';')

if not csv_exist:
    csv_lables = ["Technique", "model", "dataset", "num_errors", "num_occurrences", "num_cases_modified", "err_rate",
                  "time (s)", "word_file"]
    csv_writer.writerow(["Fairness testing"])
    csv_writer.writerow(np.concatenate([csv_lables]))
csv_writer.writerow(
    [method_name + '_' + technique, model_name, dataset_name, num_error, replaced_occurrences, changed_cases, err_rate,
     testing_time, full_lists_name])
# [csv_writer.writerow(numpy.concatenate([[words[i]], all_change_percentages[i]])) for i in range(n_words)]
csvfile.close()

# Saving Labels in csv
csv_label = output_path + "labels_count" + ".csv"
csv_label_exist = os.path.isfile(csv_label)
print("Saving Label results in {}.".format(csv_label))
csv_label_file = methods.createOpen(csv_label, 'a')
csv_label_writer = csv.writer(csv_label_file, delimiter=';')

if not csv_label_exist:
    header = ["Technique", "model", "dataset", "word_file"]
    for label in model_labels:
        header.append(label + "_More")
        header.append(label + "_Less")
    csv_label_writer.writerow(['Label counter'])
    csv_label_writer.writerow(header)

line = [method_name + '_' + technique, model_name, dataset_name, full_lists_name]
for label_num in range(num_labels):
    line.append(more_label_counter[label_num])
    line.append(less_label_counter[label_num])
csv_label_writer.writerow(line)
csv_label_file.close()

# Saving pickle with errors
details_path = output_path + 'error_details' + '/' + method_name + '_' + technique + '-' + full_lists_name + ".pkl"
print("Saving details in {}.".format(details_path))
file = methods.createOpen(details_path, "wb")
pickle.dump(changed_prediction_details, file)
file.close()

print("exit")
