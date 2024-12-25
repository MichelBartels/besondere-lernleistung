import numpy as np
from os import path

script_path = path.dirname(path.realpath(__file__))
glove = path.join(script_path, "glove.npy")
glove = np.load(glove, allow_pickle=True)
word_vectors, word_indices = glove

def get_word_embedding(word):
    if word in word_indices:
        return word_vectors[word_indices[word]]
    else:
        return np.zeros((300))

def dataset_embedding():
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
