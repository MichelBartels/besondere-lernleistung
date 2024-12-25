import random
import json
import torch
import numpy as np

def collate_fn(max_context_len, max_question_len):
    def callback(batch):
        context_len = min(max_context_len, np.amax([q[0][0].shape[-1] for q in batch]))
        question_len = min(max_question_len, np.amax([q[0][1].shape[-1] for q in batch]))
        
        contexts = np.zeros((len(batch), batch[0][0][0].shape[-2], context_len))
        questions = np.zeros((len(batch), batch[0][0][1].shape[-2], question_len))
        answers = np.zeros((len(batch), 1, context_len))

        for i, item in enumerate(batch):
            contexts[i, :, :item[0][0].shape[-1]] = item[0][0][:, :context_len]
            questions[i, :, :item[0][1].shape[-1]] = item[0][1][:, :question_len]
            answers[i, :, :item[0][0].shape[-1]] = item[1][:, :context_len]
        return (torch.from_numpy(contexts), torch.from_numpy(questions)), torch.from_numpy(answers)

    return callback

class SQuADDataset(torch.utils.data.IterableDataset):
    def __init__(self, filename, embedding, impossible_questions_possible=True, collate_fn=None, random=True, multiple_answers=False):
        self.multiple_answers = multiple_answers
        self.embedding = embedding
        self.impossible_questions_possible = impossible_questions_possible
        self.collate_fn = collate_fn
        self.random = random
        self.index = [0, 0, 0]
        with open(filename, "r") as file:
            self.data = json.load(file)["data"]
    def __next__(self):
        if self.random:
            text = random.choice(random.choice(self.data)["paragraphs"])
            question = random.choice(text["qas"])
        else:
            if self.index[2] == (len(self.data[self.index[0]]["paragraphs"][self.index[1]]["qas"]) - 1):
                if self.index[1] == (len(self.data[self.index[0]]["paragraphs"]) - 1):
                    if self.index[0] == (len(self.data) - 1):
                        return None
                    self.index[0] += 1
                    self.index[1] = 0
                    self.index[2] = 0
                else:
                    self.index[1] += 1
                    self.index[2] = 0
            else:
                self.index[2] += 1
            text = self.data[self.index[0]]["paragraphs"][self.index[1]]
            question = text["qas"][self.index[2]]
        if question["is_impossible"] and not self.impossible_questions_possible:
            return self.__next__()
        if self.impossible_questions_possible:
            embeddings = self.embedding(text["context"], question["question"], question["answers"], question["is_impossible"], multiple_answers=self.multiple_answers)
        else:
            embeddings = self.embedding(text["context"], question["question"], question["answers"], multiple_answers=self.multiple_answers)
        if embeddings != None:
            return embeddings
        return self.__next__()

    def __iter__(self):
        while True:
            item = self.__next__()
            if item == None:
                return
            yield item
