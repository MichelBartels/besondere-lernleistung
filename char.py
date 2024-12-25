import numpy as np
import string

all_chars = string.printable
embeddings_size = len(all_chars) + 1

def get_embedding_size():
    return embeddings_size

def get_char_embedding(char, index=False):
    i = all_chars.find(char)
    i = i if i != -1 else len(all_chars)
    if index:
        return i
    embedding = np.zeros((embeddings_size))
    embedding[i] = 1
    return embedding

def get_text_embeddings(text, index=False):
    if index:
        embedding = np.zeros((len(text)))
        for i, letter in enumerate(text):
            embedding[i] = get_char_embedding(text, True)
        return embedding
    embedding = np.zeros((len(text), embeddings_size))
    for i, letter in enumerate(text):
        embedding[i] = get_char_embedding(letter)
    return np.transpose(embedding, (1, 0))

def get_char_from_embedding(embedding):
    i = np.argmax(embedding)
    if i == len(all_chars):
        return " "
    return all_chars[i]

def get_text_from_embeddings(embedding):
    embedding = np.transpose(embedding, (1, 0))
    text = ""
    for i, letter in enumerate(embedding):
        text += get_char_from_embedding(letter)
    return text

def get_text_from_segmentation(text, seg):
    seg = np.transpose(seg, (1, 0))
    answer = ""
    for i in np.where(seg >= 0.5)[0]:
        answer += text[i]
    return answer

def dataset_embedding(segmentation=True):
    def callback(context, question, answers):
        x = (get_text_embeddings(context), get_text_embeddings(question))
        if segmentation:
            y = np.zeros((x[0].shape[1], 1))
            for answer in answers:
                answer_start = answer["answer_start"]
                y[answer_start:answer_start + len(answer["text"])] = 1
            return x, np.transpose(y, (1, 0))
        else:
            y = get_text_embeddings(answer["text"])
            return x, y
    return callback
