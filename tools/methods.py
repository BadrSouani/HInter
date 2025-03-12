import re
import numpy as np
import pandas as pd
import torch
import os.path
import csv
import pickle


def predict(text, tokenizer, model, isMulti, max_length, device):
    inputs = tokenizer(text, truncation=True, return_tensors='pt', max_length=max_length).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    positive = (logits >= 0.) if isMulti else (logits == max(logits[0]))
    predicted_class_id_tensor = positive[0].nonzero()  # return predicted class (multiclass)
    predicted_class_id = [predicted_class_id_tensor[i].item() for i in range(len(predicted_class_id_tensor))]
    return predicted_class_id


# Load predictions of the model from the file, or make them
def getPrediction(path, string_array, tokenizer, model, isMulti, max_length, device):
    size = len(string_array)
    prediction_file = path + "base_prediction.npy"
    if os.path.isfile(prediction_file):
        model_base_prediction = np.load(prediction_file, allow_pickle=True)
    else:
        print("Creating base predictions ...")
        model_base_prediction = [[]] * size
        for i in range(size):
            predicted_class_id = predict(string_array[i], tokenizer, model, isMulti, max_length, device)
            model_base_prediction[i] = predicted_class_id
        createDir(prediction_file)
        np.save(prediction_file, model_base_prediction)
    return model_base_prediction


def getLabelsCount(predictions, nb_label):
    labels_counter = [0] * nb_label
    for x in range(len(predictions)):
        for y in predictions[x]:
            labels_counter[y] += 1
    return labels_counter


def getTruncated(path, string_array, tokenizer, max_length):
    array_size = len(string_array)
    file_name = path + "truncated_text.pkl"
    if os.path.isfile(file_name):
        file = open(file_name, 'rb')
        truncated_text = pickle.load(file)
    else:
        print("Creating truncated text ...")
        truncated_text = [None] * array_size
        for i in range(array_size):
            encoded = tokenizer.encode(string_array[i], padding=True, truncation=True, max_length=max_length)
            truncated_text[i] = tokenizer.decode(encoded, skip_special_tokens=True)
        file = createOpen(file_name, 'wb')
        pickle.dump(truncated_text, file)
    file.close()
    return truncated_text


def getFileName(path):
    index = path.rfind("/")
    index_point = path.rfind(".")
    if index_point == -1:
        index_point = None
    return path[index + 1:index_point]

def replace(case, w_1, r_1, up_rep = False):
    up_w_1 = w_1[0].upper() + w_1[1:].lower()
    if up_rep:
        parts = r_1.split(' ')
        r_1=' '.join([x[0].upper()+x[1:] for x in parts])
    up_r_1 = r_1[0].upper() + r_1[1:]

    rep1 = re.subn(r"\b" + up_w_1 + r"\b", "[Y]", case)
    rep2 = re.subn(r"\b" + w_1 + r"\b", r_1, rep1[0], flags=re.IGNORECASE)
    return rep2[0].replace("[Y]", up_r_1), rep1[1]+rep2[1]

def createDir(path):
    index = path.rfind("/")
    if (index != -1):
        os.makedirs(path[:index], exist_ok=True)


def createOpen(path, mode):
    createDir(path)
    if 'b' in mode:
        return open(path, mode)
    else:
        return open(path, mode, newline='', encoding='utf-8')


def swap_file_words(s, x, y):
    split = s.split('_')
    for word in range(len(split)):
        if split[word] == x:
            split[word] = y
        elif split[word] == y:
            split[word] = x
    return '_'.join(split)

def format_dataset(elements):
    return [re.sub(' +', ' ', x.replace(';', '.').replace('\n', '.').replace('<br>', ' ').replace('<br/>', ' ').replace('<br />', ' ')) for x in elements]

def getMaskWords(text, tokenizer_bert, model_bert, n_prediction, device):
    inputs = tokenizer_bert(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        logits = model_bert(**inputs).logits
    mask_token_index = (inputs.input_ids == tokenizer_bert.mask_token_id)[0].nonzero(as_tuple=True)[0]
    list_logits = (logits[0, mask_token_index])[0]
    logits_sorted = list_logits.sort(descending=True)[0][0:n_prediction]
    index_words = [((list_logits == x).nonzero(as_tuple=True)[0]) for x in logits_sorted]
    return [tokenizer_bert.decode(index_words[x]) for x in range(len(index_words))]


def getFromPickle(pickle_file_path, mode):
    file = open(pickle_file_path, mode)
    file_content = pickle.load(file)
    file.close()
    return file_content

def writePickle(data, pickle_file_path, mode):
    file = createOpen(pickle_file_path, mode)
    pickle.dump(data, file)
    file.close()


def writeCSV(path, rows):
    csvfile = createOpen(path, 'w')
    csv_writer = csv.writer(csvfile, delimiter=';')
    csv_writer.writerows(rows)
    csvfile.close()


def getFiles(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            files.append(os.path.join(r, file))
    return files

def lowerPriority(seen_words, words):
    set_words = set(words)
    for seen_word in seen_words:
        if seen_word in set_words:
            words.append(words.pop(words.index(seen_word)))
    return words

def getFromCSV(path, delimiter=';', header=None):
    return pd.read_csv(path, delimiter=delimiter, header=header).to_numpy() if os.stat(path).st_size != 0 else []

def getWordlist(path):
    word_list = getFromCSV(path)
    a = word_list[:, 0] if len(word_list) != 0 else []
    b = word_list[:, 1] if len(word_list) != 0 else []
    file_name = getFileName(path)
    return file_name, a, b

def getValuesOfStringList(s):
    b_index = s.find('[')
    e_index = s.rfind(']')
    content = s[b_index+1:e_index].replace(' ', '')
    return content.split(',')

def getAnswer(chat, pipeline, size=100):
    return pipeline(
        chat,
        do_sample=False,
        num_return_sequences=1,
        max_new_tokens=size)

def getMessagePattern(sender, content):
    return {"role": sender, "content" :content,}

def getQuestion():
    return "Question: "

def getAnswerString():
    return "Answer: "

def getLabels(labels, names):
    return [names[x] for x in labels]

def getLabelsAnswer(labels, names):
    if len(labels) == 0 : return "None."
    return ", ".join(getLabels(labels, names)) + "."

def getProcessAnswer(answer, labels):
    return [x for x in range(len(labels)) if labels[x].lower() in answer.lower()]

def getText():
    return "Text: "
def getSimpleChatPattern(pairs, task = None):
    chat = []
    index = 0
    if task!=None :
        chat.append(getMessagePattern("system", task))
    while True :
        chat.append( getMessagePattern("user", getQuestion()+pairs[index][0]+"\n"+getText()+pairs[index][1]) )
        if len(pairs[index]) == 2 :
            return chat
        chat.append( getMessagePattern("assistant", "\n"+getAnswerString()+pairs[index][2]+"\n"))
        index += 1

def getLabelsInAnswer(answer, labels):
    response = []
    for l in range(len(labels)):
        if labels[l] in answer:
            response.append(l)

def getChatPattern( questions, question_pattern_a, question_pattern_b, texts, task=None):
    index = 0
    index_question = 0
    chat = []
    sender = "user"
    if task!=None :
        chat.append(getMessagePattern("system", task))
    while index != len(texts):
        if sender == "user":
            add_text = getQuestion()+question_pattern_a+questions[index_question]+question_pattern_b+"\n"+getText()+texts[index]
            index_question += 1
        else : add_text=getAnswerString()+texts[index]
        index += 1
        chat.append(getMessagePattern(sender, add_text))
        sender = "user" if sender == "assistant" else "assistant"
    return chat

def apply_chat_and_truncate(tokenizer, chat, max_input_size, output_size, end_tokens, answer_guidance, device):
    tokenized_truncated_text = tokenizer.apply_chat_template(chat,
                                                             tokenize=True,
                                                             truncation=True,
                                                             max_length=max_input_size - output_size - 10,
                                                             return_tensors="pt"
                                                             ).to(device)
    if max_input_size - output_size - 10 <= len(tokenized_truncated_text[0]): tokenized_truncated_text = torch.cat(
        (tokenized_truncated_text, end_tokens), dim=1)
    tokenized_truncated_text = torch.cat((tokenized_truncated_text, answer_guidance), dim=1).to(device)
    return tokenized_truncated_text

