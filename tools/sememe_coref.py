import os
import re

import OpenHowNet
import numpy as np
from allennlp.predictors import Predictor
import re
import torch
from tools import methods


#update version when replacement technique changes
version = "1.3.1"
#1.3.1 with dependency similarity to take similar mutants
#1.2.2 update ignore list to remove legal related occupations
#1.2.1 uses dependency to filter cases
#Has check dependent words's gender to link

male_sememe = None
female_sememe = None
occupation_sememe = None
hownet_dict = None
coreference_predictor = None
dependency_predicator = None
dependency_predicator_cpu = None
default_pos_replacement = None
dict_unk_words = None
dict_known_words = None
gender_sets = None
occupation_list = None
sentence_deleted = None

is_cuda = None

coreference_ablation = False
dependency_ablation = False
single_ablation = False
checking_ablation = False
sememe_ablation = False

MALE = 0
FEMALE = 1
NO_GENDER = 2


def init_dependency_model():
    global dependency_predicator
    global is_cuda
    is_cuda = torch.cuda.is_available()
    if dependency_predicator == None:
        dependency_predicator = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz"
            , cuda_device=0 if is_cuda else -1)

def init_dependency_model_cpu():
    global dependency_predicator_cpu
    if dependency_predicator_cpu == None:
        dependency_predicator_cpu = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz"
            , cuda_device=-1)

def init_coreference_model():
    global coreference_predictor
    if coreference_predictor == None :
        coreference_predictor = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
            , cuda_device=0 if torch.cuda.is_available() else -1)

def init_sememe_coref(male_to_female_list_param, job_list, init_models = True):
    global male_sememe, female_sememe, hownet_dict,\
        gender_sets, occupation_sememe, default_pos_replacement, dict_unk_words, dict_known_words, occupation_list,\
        sentence_deleted


    if init_models:
        init_coreference_model()
        init_dependency_model()

    #OpenHowNet.download()
    hownet_dict = OpenHowNet.HowNetDict()
    # hownet_dict.initialize_babelnet_dict()

    male_sememe = hownet_dict.get_sememe('male')[0]
    female_sememe = hownet_dict.get_sememe('female')[0]
    occupation_sememe = hownet_dict.get_sememe('Occupation')[0]

    default_pos_replacement = [{'noun':'man', 'n':'', 'v':'','pron':'him', 'Occupation':'analyst'} ,
                               {'noun':'woman', 'n':'', 'v':'', 'pron':'her', 'Occupation':'secretary'}]

    male_to_female_list_param[MALE] = [x.lower() for x in  male_to_female_list_param[MALE]]
    male_to_female_list_param[FEMALE] = [x.lower() for x in male_to_female_list_param[FEMALE]]
    job_list[MALE] = [x.lower() for x in job_list[MALE]]
    job_list[FEMALE] = [x.lower() for x in job_list[FEMALE]]
    gender_sets = []
    gender_sets.append(set(male_to_female_list_param[MALE]))
    gender_sets.append(set(male_to_female_list_param[FEMALE]))

    dict_known_words = dict(zip(male_to_female_list_param[MALE], male_to_female_list_param[FEMALE]))
    dict_known_words.update(dict(zip(male_to_female_list_param[FEMALE], male_to_female_list_param[MALE])))

    occupation_list = [set(job_list[MALE]), set(job_list[FEMALE])]

    gender_sets[MALE] = gender_sets[MALE].union(occupation_list[MALE])
    gender_sets[FEMALE] = gender_sets[FEMALE].union(occupation_list[FEMALE])

    male_to_replace = [] + [[x, default_pos_replacement[FEMALE]['Occupation']] for x in occupation_list[MALE]]
    female_to_replace = [] + [[x, default_pos_replacement[MALE]['Occupation']] for x in occupation_list[FEMALE]]

    sentence_deleted = []

    dict_known_words.update({i[0]: i[1] for i in male_to_replace})
    dict_known_words.update({i[0]: i[1] for i in female_to_replace})

    words_to_ignore = ['godparents', 'period', 'vamper', 'harrassment', 'charm', 'polygyny',
                       'cutout', 'trapes', 'skirt', 'miniskirt', 'overskirt', 'climacteric', 'maxiskirt',
                       'milk', 'sopranist', 'prodigal', 'adulterer', 'debauchee', 'coxcomb', 'oxen', 'bach',
                       'begetter', 'paramour', 'chap', 'profligate', 'voluptuary', 'almond-eyes', 'sow', 'sib',
                       'glans', 'cuss', 'angry', 'buddies', 'they', 'them', 'parents', 'appeal', 'period', 'siren',
                       'polygamy', 'lawyer', 'judge', 'clerk', 'prosecutor',
                       'debauchery', 'bunny', 'pair', 'blond', 'blonde', 'baggage',
                       'grandparents', 'erotic', 'millinery', 'virginity', 'harassment',
                       'me', 'look', 'cousin', 'milk', 'caretaker', 'gallant',
                       'peach', 'boss', 'whites', 'lulu', 'bachelor', 'baby', 'cow', 'striptease',
                       'wolf', 'molestation', 'charm', 'doll', 'milking', 'virtue', 'lover', 'grandparent',
                       'caretaker', 'sibling', 'bull', 'jenny', 'celibacy', 'child', 'fairy', 'beauty', 'couple']
    dict_unk_words = {i:[NO_GENDER,None, i] for i in words_to_ignore}

#return mutants if cache usable, else return None
#usable if same version and lists
def useMutantCache(file_path, male_to_female_list_param):
    if os.path.isfile(file_path):
        data = methods.getFromPickle(file_path, "rb")
        if data[0] == version and np.array_equal(data[1] , male_to_female_list_param):
            return data[2]
    return None

def saveMutantCache(file_path, male_to_female_list_param, mutants):
    methods.writePickle([version, male_to_female_list_param, mutants], file_path, 'wb')

#works like getWordGender, but only check the first sense
def getSimpleWordGender(original_word):
    word = original_word.lower()
    if word in dict_unk_words :
        elem = dict_unk_words[word]
        return elem[0]
    if word in gender_sets[MALE]: return MALE
    if word in gender_sets[FEMALE] : return FEMALE
    if sememe_ablation : return NO_GENDER
    sense = [x for x in hownet_dict.get_sememes_by_word(word) if x['sense'].en_grammar in default_pos_replacement[0]]
    if sense == None or len(sense) == 0: return NO_GENDER
    sense = sense[0]
    sememes = sense['sememes']
    male = male_sememe in sememes
    female = female_sememe in sememes
    if (male and female) or (not male and not female):
        dict_unk_words[word] = [NO_GENDER, None, None]
        return NO_GENDER
    elif male :
        dict_unk_words[word] = [MALE, sense, None]
        return MALE
    dict_unk_words[word] = [FEMALE, sense, None]
    return FEMALE

#Return the words of the clusters, by cluster
def getClusters(document):
    init_coreference_model()
    prediction = coreference_predictor.predict(document=document)
    clusters = prediction['clusters']

    for c1 in range(0, len(clusters)):
        for c2 in range(c1+1, len(clusters)):
            for tuple1 in clusters[c1]:
                for tuple2 in clusters[c2]:
                    if not tuple1[0] > tuple2[1] and not tuple2[0] > tuple1[1]:
                        if tuple1[1]-tuple1[0] > tuple2[1]-tuple2[0]:
                            a = tuple1
                            b = tuple2
                        else:
                            a = tuple2
                            b = tuple1
                        if a[0] >= b[0]:
                            a[0] = b[1]+1
                        else:
                            a[1] = b[0]-1
    num_clusters = []
    for id in range(len(clusters)):
        num_cluster = []
        for inter in clusters[id]:
            [num_cluster.append(x) for x in list(range(inter[0], inter[1]+1))]
        num_clusters.append([id,num_cluster])
    return prediction, num_clusters

def getClusterGender(doc, cluster):
    for group_index in range(1, len(cluster)):
        temp_male_cluster = []
        temp_female_cluster = []
        for x in cluster[group_index]:
            word = doc[x]
            if word in gender_sets[MALE]: temp_male_cluster.append(x)
            elif word in gender_sets[FEMALE]: temp_female_cluster.append(x)
            else:
                word_gender = getSimpleWordGender(word)
                if word_gender == MALE : temp_male_cluster.append(x)
                elif word_gender == FEMALE : temp_female_cluster.append(x)

        len_male = len(temp_male_cluster)
        len_female = len(temp_female_cluster)
        if len_male != 0 or len_female != 0:
            if len_male > len_female : return MALE
            else : return FEMALE
        return NO_GENDER

#Return male_clusters and female_clusters words with unique id by entity
def getClustersByGender(doc, clusters):
    male_clusters = []
    female_clusters = []
    #for each cluster, define if the target is a male or a female
    for cluster in clusters:
        id = cluster[0]
        groups = []
        gender = -1
        for group_index in range(1, len(cluster)):
            temp_male_cluster = []
            temp_female_cluster = []
            #define the gender of each word for the target, and use the majority
            # to define target's gender and keep this gender's words
            for x in cluster[group_index]:
                word = doc[x]
                if word in gender_sets[MALE]: temp_male_cluster.append(x)
                elif word in gender_sets[FEMALE]: temp_female_cluster.append(x)
                else:
                    word_gender = getSimpleWordGender(word)
                    if word_gender == MALE : temp_male_cluster.append(x)
                    elif word_gender == FEMALE : temp_female_cluster.append(x)

            len_male = len(temp_male_cluster)
            len_female = len(temp_female_cluster)
            if len_male != 0 or len_female != 0:
                if len_male > len_female : groups += [temp_male_cluster]; gender = MALE
                else : groups += [temp_female_cluster]; gender = FEMALE
        if len(groups)==0:continue
        if gender == MALE: male_clusters += [[id] + groups]
        else : female_clusters += [[id] + groups]
    return male_clusters, female_clusters

def getClustersEntities(doc, clusters):
    to_ignore_words = set(["the", "an", 'a', ',', '!', '?', ',', '.', ';', '\'', 'this', 'their', 'those', 'these', 'whom'])
    new_clusters = []
    for cluster in clusters:
        id = cluster[0]
        used_index = [i for i in cluster[1] if doc[i] not in to_ignore_words]
        concat_words = []
        words = []
        gender = False
        for i in range(len(used_index)):
            x = used_index[i]
            word = doc[x]
            if gender == False:
                if word in gender_sets[MALE]: gender = True
                elif word in gender_sets[FEMALE]: gender = True
                else:
                    word_gender = getSimpleWordGender(word)
                    if word_gender == MALE : gender = True
                    elif word_gender == FEMALE : gender = True
            if i != 0 and used_index[i-1]+1 == used_index[i]:
                concat_words[-1] += " "+word
            else:
                concat_words.append(word)
            words.append(word)
        if gender :
            new_clusters.append([id, concat_words, words])
    return new_clusters

def split_into_sentences(text):
    #https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences
    alphabets = "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"
    digits = "([0-9])"
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

def untokenize(words):
    #https://github.com/rizkiarm/LipNet/blob/master/lipnet/utils/spell.py
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
         "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()

#return word by which changing the word of the tuple. First checked in lists,
# else added to unknown words and returning default. If word has no default, add it to the dict and return None
def getWordReplacement(word_index, tokens):
    word = tokens[word_index].lower()
    word_gender = getSimpleWordGender(word)

    if word in ['his', 'her', 'hers', 'him']:
        # if word == "his": rpl1="her"; rpl2 = "her's"
        if word == "his" or word == "him": return "her"
        if word == "hers": return "his"
        if word == "her": rpl1="his"; rpl2 = "him"
        new_tokens = tokens.copy()
        new_tokens[word_index] = '[MASK]'
        sentences = split_into_sentences(untokenize(new_tokens))
        target_sentence = [s for s in range(len(sentences)) if '[MASK]' in sentences[s]]
        if len(target_sentence) > 0:
            index_sentence_with_mask = target_sentence[0]
            sentence = capitalize(sentences[index_sentence_with_mask])
            sentence1 = sentence.replace('[MASK]', rpl1)
            sentence2 = sentence.replace('[MASK]', rpl2)

            original_sentences = split_into_sentences(untokenize(tokens))
            dp_o = dependency_predicator.predict(capitalize(original_sentences[index_sentence_with_mask]))
            pos_dp_o = dp_o['pos']
            sentence_start = 0
            for x in range(index_sentence_with_mask): sentence_start+=len(dependency_predicator._tokenizer.tokenize(original_sentences[x]))
            index_word_sentence = word_index - sentence_start
            if index_word_sentence+1 < len(pos_dp_o) and dp_o['words'][index_word_sentence].lower() == word:
                pos_next_word = pos_dp_o[index_word_sentence+1]
                if pos_next_word in ['ADP', 'PART', 'SCONJ']: return rpl2
                if tokens[index_word_sentence+1] != '"' and pos_next_word == 'PUNCT' : return rpl2
                if index_word_sentence+2 < len(pos_dp_o):
                    if pos_next_word == 'CCONJ' and not (tokens[index_word_sentence+1] == 'or' and tokens[index_word_sentence+2] =='her'): return rpl2
                if pos_next_word in ['NOUN', 'ADJ']: return rpl1

            dp_1 = dependency_predicator.predict(sentence1)
            dp_2 = dependency_predicator.predict(sentence2)


            tokens_dp_0 = dp_o['words']
            tokens_dp_1 = dp_1['words']

            return rpl1 if dp_1['loss'] < dp_2['loss'] else rpl2

    if word in dict_known_words: return dict_known_words[word]
    word_info = dict_unk_words[word]
    sense = word_info[1]
    word_rpl = word_info[2]
    if word_rpl != None : return word_rpl
    else :
        word_type = 'Occupation' if occupation_sememe in sense['sememes'] else sense['sense'].en_grammar
        word_rpl = default_pos_replacement[1-word_gender].get(word_type, None)
        dict_unk_words[word] = [word_gender, sense, word_rpl]
        return word_rpl

#return a list with the mutants for the replacement words. words_gender to MALE if made of male words, else FEMALE
def getClusterReplacements(cluster, tokens):
    result = []
    for group_index in range(1, len(cluster)):
        result += [[getWordReplacement(index, tokens) for index in cluster[group_index]]]
    return [cluster[0]] + result
def getClustersReplacements(male_clusters, female_clusters, tokens):
    male_clusters_replacement = []
    female_clusters_replacement = []
    for cluster in male_clusters:
        male_clusters_replacement.append(getClusterReplacements(cluster, tokens))
    for cluster in female_clusters:
        female_clusters_replacement.append(getClusterReplacements(cluster, tokens))
    return male_clusters_replacement, female_clusters_replacement

def single_replacement(coref_prediction, num_clusters):
    original = coref_prediction['document']
    male_to_female = []
    female_to_male = []
    ignored_words = set()
    [[[ignored_words.add(x) for x in cluster[group]] for group in range(1, len(cluster))] for cluster in num_clusters]
    for x in range(0, len(original)):
        if x in ignored_words: continue
        word_gender = getSimpleWordGender(original[x])
        if word_gender == NO_GENDER : continue
        replacement = getWordReplacement(x, original)
        if replacement == default_pos_replacement[1-word_gender]['Occupation']:
            for occupation in occupation_list[1-word_gender]:
                new_case = original.copy()
                new_case[x] = occupation
                joined = untokenize(new_case)
                if word_gender == MALE:male_to_female.append([joined, [[original[x], new_case[x]]]])
                else:female_to_male.append([joined, [[original[x], new_case[x]]]])
        else:
            new_case = original.copy()
            if new_case[x][0].isupper() : new_case[x] = capitalize(replacement)
            else : new_case[x] = replacement
            joined = untokenize(new_case)
            if word_gender == MALE: male_to_female.append([joined, [[original[x], new_case[x]]]])
            else : female_to_male.append([joined, [[original[x], new_case[x]]]])
    return male_to_female, female_to_male



def coref_depen_replacement(prediction, num_clusters, male_clusters, male_replacement, female_clusters, female_replacement):
    original = prediction['document']

    male_to_female = []
    female_to_male = []
    clusters = [male_clusters, female_clusters]
    replacements = [male_replacement, female_replacement]
    cases = []
    for ind in [MALE, FEMALE]:
        for x in range(0, len(clusters[ind])):
            target = clusters[ind][x]
            id = target[0]
            replacement_list = [l for l in replacements[ind] if l[0] == id]
            for replacement in replacement_list:
                modifications = []
                tokens = original.copy()
                for group_index in range(1, len(target)):
                    modification = []
                    group = target[group_index]
                    replacement_target = replacement[group_index]
                    word_nb = 0
                    word_replacement = replacement_target[word_nb]
                    nb_words = len(group)
                    word = original[target[group_index][word_nb]]
                    for index in num_clusters[id][group_index]:
                        if tokens[index] == word:
                            if tokens[index][0].isupper() : tokens[index] = capitalize(word_replacement)
                            else : tokens[index] = word_replacement
                            word_nb += 1
                            modification.append([word, word_replacement])
                            if word_nb >= nb_words: break
                            word = original[target[group_index][word_nb]]
                            word_replacement = replacement_target[word_nb]
                        if word_nb >= nb_words: break
                    modifications += [modification]
                cases.append([ind, untokenize(tokens), modifications[0], modifications[1] if len(modifications)>1 else []])

    return cases

def mutant_from_tokens(original_text, original_tokens, mutant_tokens, modified_tokens):
    if len(original_tokens) != len(mutant_tokens) :
        print("Error, different number of tokens in original and mutant.")
        return None
    modified_tokens.sort()
    index=0
    mutant = ""
    position_original = 0
    while index != len(original_tokens) and len(modified_tokens) != 0:
        rplt = ""
        if index == modified_tokens[0]:
            rplt = mutant_tokens[index]
            modified_tokens.pop(0)
            # remove index from array, leave if array empty by adding rest of original. Add uppercase if present (use methods)
        original_word = original_tokens[index]
        position_in_text = original_text.find(original_word, position_original)
        end_word = position_in_text + len(original_word)
        used_part = original_text[position_original:end_word]
        position_original += len(used_part)
        new_part = used_part if rplt=="" else methods.replace(used_part, original_word, rplt)[0]
        mutant += new_part
        index += 1

    if len(modified_tokens) != 0 :
        print("Error, all words were not replaced.")
        return None
    mutant+= original_text[position_original:]
    return mutant

def coref_depen_replacement2(original_text, sentences, prediction, num_clusters, male_clusters, male_replacement, female_clusters, female_replacement):
    original_tokens = prediction['document']
    clusters = [male_clusters, female_clusters]
    replacements = [male_replacement, female_replacement]
    cases = []
    for ind in [MALE, FEMALE]: #For Male/Female replacement
        for x in range(0, len(clusters[ind])): #For each entity to replace
            target = clusters[ind][x]
            id = target[0]
            replacement_list = [l for l in replacements[ind] if l[0] == id]
            modified_tokens = []
            for replacement in replacement_list:
                modifications = []
                tokens = original_tokens.copy()
                for group_index in range(1, len(target)):
                    modification = []
                    group = target[group_index]
                    replacement_target = replacement[group_index]
                    word_nb = 0
                    word_replacement = replacement_target[word_nb]
                    nb_words = len(group)
                    word = original_tokens[target[group_index][word_nb]]
                    for index in num_clusters[id][group_index]:
                        if tokens[index] == word:
                            tokens[index] = word_replacement
                            modified_tokens.append(index)
                            word_nb += 1
                            modification.append([word, word_replacement])
                            if word_nb >= nb_words: break
                            word = original_tokens[target[group_index][word_nb]]
                            word_replacement = replacement_target[word_nb]
                        if word_nb >= nb_words: break
                    modifications += [modification]
                mutant_case = mutant_from_tokens(original_text, original_tokens, tokens, modified_tokens)
                if mutant_case == None : continue
                cases.append([ind, mutant_case,modifications[0], modifications[1] if len(modifications)>1 else []])

    return cases

def smart_table_comparison(table1, table2):
    x = 0
    y = 0
    error_limit = abs(len(table1) - len(table2))
    error = 0
    shift = 0
    while x < len(table1) and y < len(table2):
        if table1[x] != table2[y]:
            error +=1
            if shift < error_limit :
                shift += 1
                if len(table1) > len(table2): x += 1
                elif len(table1) < len(table2): y += 1
        x += 1
        y += 1
    error += (len(table1)-x) + (len(table2)-y)
    return error <= error_limit, error, max(len(table1), len(table2))-error


def isSentenceStructureSimilar(prediction1, prediction2, strict = True):
    similar = True
    pos_comparison = smart_table_comparison(prediction1['pos'], prediction2['pos'])
    if not pos_comparison[0]: similar = False
    pred_comparison = True, 0, 0
    if strict :
        pred_dep1 = prediction1['predicted_dependencies']
        pred_dep2 = prediction2['predicted_dependencies']
        pred_comparison = smart_table_comparison(pred_dep1, pred_dep2)
        if not pred_comparison[0]: similar = False
    return similar, pos_comparison, pred_comparison


def dependency(text, clusters, coref_prediction):
    init_dependency_model()
    original = coref_prediction['document']
    sentences = split_into_sentences(text)
    sentences_tokenized = [dependency_predicator._tokenizer.tokenize(s) for s in sentences]
    dependency_untokenized = "".join([" ".join([t.text for t in s]) for s in sentences_tokenized])
    ##word start in text
    indices = []
    last_indice = 0
    for word in original:
        indice = dependency_untokenized.find(word, last_indice)
        indices.append(indice)
        last_indice = indice
    ####################

    token_index = 0
    sentence_token_start = 0
    ignored_words = set()
    [[ignored_words.add(x) for x in cluster[1] ]for cluster in clusters]

    sentence_dependency = [None] * len(sentences)
    dependent_words = [[x,[]] for x in range(len(clusters))]

    for x in range(len(sentences_tokenized)):
        s_t = sentences_tokenized[x]
        sentence_capitalized = capitalize(sentences[x])
        for t in s_t:
            if token_index == len(original) :
                print('Error : tokenized sentences different.');break
            if token_index in ignored_words: token_index+=1; continue
            word_gender = getSimpleWordGender(original[token_index])
            if word_gender == NO_GENDER: token_index+=1; continue
            if sentence_dependency[x] == None: sentence_dependency[x] = dependency_predicator.predict(sentence = sentence_capitalized)
            id = getWordReferenceDependency(t.text, indices[sentence_token_start], clusters, sentence_dependency[x], indices)
            if id == -1 : token_index += 1; continue
            if getClusterGender(original, clusters[id]) != word_gender: token_index += 1; continue
            mutant_token = [w.text for w in s_t]
            rpl_word = getWordReplacement(token_index, original)
            mutant_token[token_index-sentence_token_start] = rpl_word
            mutant_sentence = capitalize(untokenize(mutant_token))
            dp_mutant = dependency_predicator.predict(sentence = mutant_sentence)
            struct_similar = isSentenceStructureSimilar(sentence_dependency[x], dp_mutant)
            if not struct_similar[0] :
                sentence_deleted.append([mutant_sentence, [[], [t.text, rpl_word]]])
                token_index +=1; continue
            ####################################
            dependent_words[id][1].append(token_index)
            token_index += 1
        sentence_token_start+=len(s_t)
    return dependent_words

def dependency2(text, clusters, coref_prediction):
    init_dependency_model()
    original = coref_prediction['document']
    sentences = split_into_sentences(text)
    sentences_tokenized = [dependency_predicator._tokenizer.tokenize(s) for s in sentences]
    dependency_untokenized = "".join([" ".join([t.text for t in s]) for s in sentences_tokenized])
    ##word start in text
    indices = []
    last_indice = 0
    for word in original:
        indice = dependency_untokenized.find(word, last_indice)
        indices.append(indice)
        last_indice = indice
    ####################

    token_index = 0
    sentence_token_start = 0
    ignored_words = set()
    [[ignored_words.add(x) for x in cluster[1] ]for cluster in clusters]

    sentence_dependency = [None] * len(sentences)
    dependent_words = [[x,[]] for x in range(len(clusters))]

    for x in range(len(sentences_tokenized)):
        s_t = sentences_tokenized[x]
        sentence_capitalized = capitalize(sentences[x])
        for t in s_t:
            if token_index == len(original) :
                print('Error : tokenized sentences different.');break
            if token_index in ignored_words: token_index+=1; continue
            word_gender = getSimpleWordGender(original[token_index])
            if word_gender == NO_GENDER: token_index+=1; continue
            if sentence_dependency[x] == None: sentence_dependency[x] = dependency_predicator.predict(sentence = sentence_capitalized)
            id = getWordReferenceDependency(t.text, indices[sentence_token_start], clusters, sentence_dependency[x], indices)
            if id == -1 : token_index += 1; continue
            if getClusterGender(original, clusters[id]) != word_gender: token_index += 1; continue
            dependent_words[id][1].append(token_index)
            token_index += 1
        sentence_token_start+=len(s_t)
    return dependent_words

def capitalize(text):
    for x in range(len(text)):
        if text[x].isalpha():
            return text[:x] + text[x].upper() + text[x+1:]
    return text

#Return a list with the nodes the word is directly dependent to
def getWordLinks(word, parent, tree):
    if tree['word'].lower() == word.lower():
        nodes = []
        if parent != None : nodes.append(parent)
        if 'children' in tree: nodes += tree['children']
        return nodes
    if 'children' in tree:
        for children in tree['children']:
            links = getWordLinks(word, tree, children)
            if len(links)>0: return links
    return []

#return the id of the entity the reference points to, if it is one or has one.
# Else return -1
def getNodeReferenceId(node, sentence_char_start, clusters, indices):
    word_index = sentence_char_start + node['spans'][0]['start']
    for cluster in clusters:
        for index in cluster[1]:
            if indices[index] == word_index:
                return cluster[0]
    return -1

#Return the id of the entity the word is dependent to
#If is not, return -1
def getWordReferenceDependency(word, sentence_start, clusters, depen_prediction, token_char_indices):
    node = depen_prediction['hierplane_tree']['root']
    links = getWordLinks(word, None, node) #get words linked to our word
    for link in links:
        id = getNodeReferenceId(link, sentence_start,clusters, token_char_indices)
        if id != -1 :
            return id
    return -1



def isTextStructureSimilar(original_sentences, original_dependency, mutant_sentences, strict = True, limit_sentence_length=500):
    init_dependency_model()
    similar = True
    pos_list = []
    dep_list = []
    if len(original_sentences) != len(mutant_sentences) :
        return False, original_dependency, None, None, 1
    for i in range(len(original_sentences)):
        if original_sentences[i] == mutant_sentences[i] : continue

        if len(mutant_sentences[i]) > limit_sentence_length:
            return False, original_dependency, None, None, 1

        mut_dep = dependency_predicator.predict(sentence=capitalize(mutant_sentences[i]))
        if original_dependency[i] == None:
            original_dependency[i] = dependency_predicator.predict(sentence = capitalize(original_sentences[i]))

        similarity = isSentenceStructureSimilar(original_dependency[i], mut_dep, strict)
        if not similarity[0]: similar = False
        pos_list.append(similarity[1])
        dep_list.append(similarity[2])
    return similar, original_dependency, pos_list, dep_list, 0


#return all the mutants generated from a case, with for each case :
#0#male_female : MALE if male to female else FEMALE
#1#modified : case after mutation
#2#[coreference modifications] : array of the replacement pairs used by using coreference [word, replacement]
#3#[dependency modifications] : array of the replacement pairs used by using dependency parsing [word, replacement]
#4#[atomic modifications]
#5#nb dictionary: nb words detected by dictionary
#6#nb sememe: words detected by sememes
#7#similarity check : if the mutant passed the similarity check 1 or not 0 or ablation -1
#8#error : if the mutant gives an error
#9#case : case as it was used as input by the model
#10#case number : id of the case
def getSentenceReplacements(original_text):
    #Ensure good separation. For better coreference and dependency
    sentences = split_into_sentences(original_text)
    if len(sentences) == 0 : return None
    text = ' '.join(sentences)
    while '  ' in text: text = text.replace("  ", " ") #remove multiple spaces
    if len(dependency_predicator._tokenizer.tokenize(text)) < 3: return None #text too short, not usable
    prediction, clusters = getClusters(text)

    #Dependency pasing to find dependent words
    if not dependency_ablation:
        clusters_dependency = dependency(text, clusters, prediction)
        #union of coreference and dependency words to replace
        for dep_cluster in clusters_dependency:
            id = dep_cluster[0]
            for cluster in clusters:
                if cluster[0] == id and len(dep_cluster[1]) > 0:
                    cluster.append(dep_cluster[1])
                    break
    ###################################################

    male_clusters, female_clusters = getClustersByGender(prediction['document'], clusters)
    male_replacements, female_replacements = getClustersReplacements(male_clusters, female_clusters, prediction['document'])
    #Single (atomic) replacement of not mutated words
    if not single_ablation:
        male_single, female_single = single_replacement(prediction, clusters)
    else: male_single, female_single = [], []

    #multiple replacement for jobs
    for l in [[male_replacements, female_replacements], [male_single, female_single]] :
        for i in [MALE, FEMALE]:
            gender_list = l[i]
            temp = []
            for mutant in gender_list:
                for group in range(1,len(mutant)):
                    modifications = mutant[group]
                    for m in range(len(modifications)):
                        if modifications[m] == default_pos_replacement[1-i]['Occupation']:
                            for occupation in occupation_list[1-i]:
                                if occupation != modifications[m]:
                                    new_mutant = mutant.copy()
                                    new_mutant[group] = modifications.copy()
                                    new_mutant[group][m] = occupation
                                    temp.append(new_mutant)
            l[i] += temp


    mutants = coref_depen_replacement(prediction, clusters, male_clusters, male_replacements, female_clusters, female_replacements)

    formated_mutants = []
    for c in mutants:
        nb_dictionary = 0
        nb_sememe = 0
        for m in [c[2],c[3]]:
            for p in m:
                if p[0].lower() in gender_sets[c[0]]: nb_dictionary+=1
                else : nb_sememe+=1
        formated_mutants.append(c+[[], nb_dictionary, nb_sememe])
    for t in [[male_single, MALE], [female_single, FEMALE]]:
        for c in t[0]:
            nb_dictionary = 0
            nb_sememe = 0
            if c[1][0][0].lower() in gender_sets[t[1]]: nb_dictionary+=1
            else : nb_sememe+=1
            formated_mutants.append([t[1], c[0], [], [], [c[1][0]], nb_dictionary, nb_sememe])

    # remove mutants that have a structure different from the original text
    if not checking_ablation:
        sentences = split_into_sentences(text)
        sentence_dependency = [None] * len(sentences)
        for l in formated_mutants:
            isSimilar, sentence_dependency , pos_list, dep_list, invalid = isTextStructureSimilar(sentences,
                                                        sentence_dependency, split_into_sentences(l[1]))
            l += [1 if isSimilar else 0]
    else:
        for l in formated_mutants:
            l += [-1]
    #######################################################################

    return formated_mutants


def getLongSentenceReplacements(original_text):
    sentences = split_into_sentences(original_text)
    if len(sentences) == 0 : return None
    text = original_text

    prediction, clusters = getClusters(text)

    #Dependency pasing to find dependent words
    if not dependency_ablation:
        clusters_dependency = dependency2(text, clusters, prediction)
        #union of coreference and dependency words to replace
        for dep_cluster in clusters_dependency:
            id = dep_cluster[0]
            for cluster in clusters:
                if cluster[0] == id and len(dep_cluster[1]) > 0:
                    cluster.append(dep_cluster[1])
                    break
    ###################################################

    male_clusters, female_clusters = getClustersByGender(prediction['document'], clusters)
    male_replacements, female_replacements = getClustersReplacements(male_clusters, female_clusters, prediction['document'])
    #Single (atomic) replacement of not mutated words
    if not single_ablation:
        male_single, female_single = single_replacement(prediction, clusters)
    else: male_single, female_single = [], []

    #multiple replacement for jobs
    for l in [[male_replacements, female_replacements], [male_single, female_single]] :
        for i in [MALE, FEMALE]:
            gender_list = l[i]
            temp = []
            for mutant in gender_list:
                for group in range(1,len(mutant)):
                    modifications = mutant[group]
                    for m in range(len(modifications)):
                        if modifications[m] == default_pos_replacement[1-i]['Occupation']:
                            for occupation in occupation_list[1-i]:
                                if occupation != modifications[m]:
                                    new_mutant = mutant.copy()
                                    new_mutant[group] = modifications.copy()
                                    new_mutant[group][m] = occupation
                                    temp.append(new_mutant)
            l[i] += temp


    mutants = coref_depen_replacement2(original_text, sentences, prediction, clusters, male_clusters, male_replacements, female_clusters, female_replacements)

    formated_mutants = []
    for c in mutants:
        nb_dictionary = 0
        nb_sememe = 0
        for m in [c[2],c[3]]:
            for p in m:
                if p[0].lower() in gender_sets[c[0]]: nb_dictionary+=1
                else : nb_sememe+=1
        formated_mutants.append(c+[[], nb_dictionary, nb_sememe])
    for t in [[male_single, MALE], [female_single, FEMALE]]:
        for c in t[0]:
            nb_dictionary = 0
            nb_sememe = 0
            if c[1][0][0].lower() in gender_sets[t[1]]: nb_dictionary+=1
            else : nb_sememe+=1
            formated_mutants.append([t[1], c[0], [], [], [c[1][0]], nb_dictionary, nb_sememe])

    # remove mutants that have a structure different from the original text
    if not checking_ablation:
        sentences = split_into_sentences(text)
        sentence_dependency = [None] * len(sentences)
        for l in formated_mutants:
            isSimilar, sentence_dependency , pos_list, dep_list, invalid = isTextStructureSimilar(sentences,
                                                        sentence_dependency, split_into_sentences(l[1]))
            l += [1 if isSimilar else 0]
    else:
        for l in formated_mutants:
            l += [-1]
    #######################################################################

    return formated_mutants


