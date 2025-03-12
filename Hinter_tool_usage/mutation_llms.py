import torch
import time
import methods
import utils as sm

def fct_modification(fct_replacement, word_list, complete_dataset_size, dataset_text, replacement_list=[]):
    discarded_mutants_length = 0
    dependencies = [None] * complete_dataset_size
    original_sentences_array = [None] * complete_dataset_size
    l_replaced_occurrences = 0
    mutants = []
    for i_word in range(len(word_list)):
        #print("Pair ", i_word+1, "/", len(word_list))
        word = word_list[i_word]
        replacement_word = fct_replacement(i_word, replacement_list)
        for i in range(complete_dataset_size):
            modified_text, occurrences = methods.replace(dataset_text[i], word, replacement_word, False)
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


def fct_intersectional(word_list_1, replacement_list_1, word_list_2,replacement_list_2,complete_dataset_size, dataset_text):
    discarded_mutants_length = 0
    dependencies = [None]*complete_dataset_size
    original_sentences_array = [None]*complete_dataset_size
    mutants = []
    absence = set()
    for i in range(complete_dataset_size):
        #print("Doing ", i + 1, "/", complete_dataset_size)
        for i_word in range(len(word_list_1)):
            word_1 = word_list_1[i_word]
            replacement_word_1 = replacement_list_1[i_word]
            key1 = tuple([i, word_1])
            if key1 in absence: continue
            modified_text_w1, occurrences = methods.replace(dataset_text[i], word_1, replacement_word_1, False)
            if occurrences == 0:
                absence.add(key1)
                continue
            for y_word in range(len(word_list_2)):
                word_2 = word_list_2[y_word]
                key2 = tuple([i, word_2])
                if key2 in absence: continue
                if word_1 == word_2 or word_2 == replacement_word_1: continue
                replacement_word_2 = replacement_list_2[y_word]
                modified_text_w1_w2, occurrences2 = methods.replace(modified_text_w1, word_2, replacement_word_2, False)
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

    return 0, mutants, discarded_mutants_length


def fct_replacement(list_word, list_replacement,  complete_dataset_size, dataset_text,):
    def replacement(it, list_of_word): return list_of_word[it]

    return fct_modification(replacement, list_word,  complete_dataset_size, dataset_text, list_replacement)


def mutation(dataset_text, dict_path, inter_dict_path=""):
    word_file_name, new_word_list_A, new_word_list_B = methods.getWordlist(dict_path)
    if inter_dict_path != "":
        inter_file_name, inter_list_A, inter_list_B = methods.getWordlist(inter_dict_path)
        is_intersectional = True
    else:
        is_intersectional = False

    complete_dataset_size = len(dataset_text)

    start_time = time.time()

    if is_intersectional:
        occurrences, mutants_list, discarded = fct_intersectional(new_word_list_A, new_word_list_B,
                                                             inter_list_A, inter_list_B, complete_dataset_size, dataset_text)
    else:
        occurrences, mutants_list, discarded = fct_replacement(new_word_list_A, new_word_list_B , complete_dataset_size, dataset_text)

    generation_time = time.time() - start_time

    if is_intersectional:
        header = ['Word 1', 'Replacement 1', 'Word 2', 'Replacement 2', 'Mutant', 'Index', 'Original', "Similarity"]
    else:
        header = ['Word', 'Replacement', 'Mutant', 'Index', 'Original', "Occurrences", "Similarity"]

    dict_results = {"time (s)": generation_time, "occurrences": occurrences,
                    "header": header, 'discarded_test_cases_sentence': discarded}
    lines = [dict_results, mutants_list]

    #print(f"Done generating {len(mutants_list)} mutants for {generation_time} seconds.")
    return lines

#ex1 = mutation(["The man.", "The black man."], "../data/gender/male_female.csv")
#ex2 = mutation(["The man.", "The black man."], "../data/race/minority_majority.csv", "../data/gender/male_female.csv")

