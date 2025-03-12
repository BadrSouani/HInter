import re
from allennlp.predictors import Predictor
import torch

dependency_predicator = None

def split_into_sentences(text):
    #https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences
    alphabets = "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\\s|She\\s|It\\s|They\\s|Their\\s|Our\\s|We\\s|But\\s|However\\s|That\\s|This\\s|Wherever)"
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
    text = re.sub("\\s" + alphabets + "[.] "," \\1<prd> ",text)
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

def isSentenceStructureSimilar(prediction1, prediction2, strict=True):
    def smart_table_comparison(table1, table2):
        x, y, error_limit = 0, 0, abs(len(table1) - len(table2))
        error, shift = 0, 0
        while x < len(table1) and y < len(table2):
            if table1[x] != table2[y]:
                error += 1
                if shift < error_limit:
                    shift += 1
                    if len(table1) > len(table2): x += 1
                    elif len(table1) < len(table2): y += 1
            x += 1
            y += 1
        error += (len(table1) - x) + (len(table2) - y)
        return error <= error_limit, error, max(len(table1), len(table2)) - error

    similar = True
    pos_comparison = smart_table_comparison(prediction1['pos'], prediction2['pos'])
    if not pos_comparison[0]: similar = False
    pred_comparison = True, 0, 0
    if strict:
        pred_dep1, pred_dep2 = prediction1['predicted_dependencies'], prediction2['predicted_dependencies']
        pred_comparison = smart_table_comparison(pred_dep1, pred_dep2)
        if not pred_comparison[0]: similar = False
    return similar, pos_comparison, pred_comparison

def isTextStructureSimilar(original_sentences, original_dependency, mutant_sentences, strict=True, limit_sentence_length=500):
    def init_dependency_model():
        global dependency_predicator
        if 'dependency_predicator' not in globals():
            dependency_predicator = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz",
                cuda_device=0 if torch.cuda.is_available() else -1
            )

    init_dependency_model()
    similar, pos_list, dep_list = True, [], []
    if len(original_sentences) != len(mutant_sentences):
        return False, original_dependency, None, None, 1

    for i in range(len(original_sentences)):
        if original_sentences[i] == mutant_sentences[i]: continue

        if len(mutant_sentences[i]) > limit_sentence_length:
            return False, original_dependency, None, None, 1

        mut_dep = dependency_predicator.predict(sentence=mutant_sentences[i].capitalize())
        if original_dependency[i] is None:
            original_dependency[i] = dependency_predicator.predict(sentence=original_sentences[i].capitalize())

        similarity = isSentenceStructureSimilar(original_dependency[i], mut_dep, strict)
        if not similarity[0]: similar = False
        pos_list.append(similarity[1])
        dep_list.append(similarity[2])

    return similar, original_dependency, pos_list, dep_list, 0
