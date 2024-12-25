import json
import random
import itertools
from os import path
import multiprocessing
from collections import deque
import random
from transformers import BertTokenizerFast

class SQuAD:
    def __init__(self, batch_size, test=False, shuffle=True, dummy=False):
        self.path = path.dirname(__file__)
        with open(path.join(self.path, "data/dev-v2.0.json" if test != False else "data/train-v2.0.json"), "r") as file:
            self.data = json.loads(file.read())["data"]
        self.paragraphs = []
        self.positive_pairs = {}
        self.negative_pairs = {}
        for article in self.data:
            for paragraph in article["paragraphs"]:
                paragraph_str = paragraph["context"]
                self.paragraphs.append(paragraph_str)
                self.positive_pairs[paragraph_str] = []
                self.negative_pairs[paragraph_str] = []
                for qa in paragraph["qas"]:
                    question = qa["question"]
                    if qa["is_impossible"]:
                        self.negative_pairs[paragraph_str].append(question)
                    else:
                        self.positive_pairs[paragraph_str].append(question)

        self.examples = list(self.__load_examples(dummy))
        self.batch_size = batch_size
        self.conn = False
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def __load_examples(self, dummy=False):
        for i, paragraph in enumerate(self.paragraphs):
            positives = self.positive_pairs[paragraph]
            negatives = self.negative_pairs[paragraph]
            for positive, negative in itertools.zip_longest(positives, negatives):
                if positive == None:
                    if len(positives) == 0:
                        continue
                    else:
                        positive = random.choice(positives)
                while negative == None:
                    index = random.randrange(0, len(self.paragraphs))
                    if index == i:
                        continue
                    negative_paragraph = self.paragraphs[index]
                    if len(self.negative_pairs[negative_paragraph]) == 0:
                        continue
                    negative = random.choice(self.negative_pairs[negative_paragraph])
                yield [paragraph, positive, negative]
                if dummy:
                    return

    def __iter__(self):
        if self.conn:
            self.conn.send(False)
        def bg_generator(conn, examples, batch_size, tokenizer):
            indices = list(range(len(examples)))
            random.shuffle(indices)
            indices = deque(indices)
            while conn.recv():
                batch = [] 
                while len(indices) != 0 and len(batch) != batch_size:
                    batch.append(examples[indices.pop()])
                if len(batch) == 0:
                    conn.send(None)
                    return
                paragraphs, positives, negatives = tuple(zip(*batch))
                paragraphs = tokenizer.batch_encode_plus(list(paragraphs), padding=True, max_length=400, return_tensors="pt")
                positives = tokenizer.batch_encode_plus(list(positives), padding=True, max_length=30, return_tensors="pt")
                negatives = tokenizer.batch_encode_plus(list(negatives), padding=True, max_length=30, return_tensors="pt")
                conn.send((paragraphs, positives, negatives))
        self.conn, child_conn = multiprocessing.Pipe()
        p = multiprocessing.Process(target=bg_generator, args=(child_conn, self.examples, self.batch_size, self.tokenizer))
        p.start()
        self.conn.send(True)
        return self

    def __len__(self):
        remainder = len(self.examples) % self.batch_size
        length = (len(self.examples) - remainder) / self.batch_size
        if remainder > 0:
            length += 1
        return int(length)

    def __next__(self):
        batch = self.conn.recv()
        if batch == None:
            raise StopIteration
        self.conn.send(True)
        return batch
