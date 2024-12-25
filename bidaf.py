from ..model import Net
import nltk
from ...embeddings.glove import get_word_embedding
from ...embeddings.char import get_text_embeddings as get_char_embeddings
from ...embeddings.char import get_embedding_size as get_char_embedding_size
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class BiDAF(Net):
    def __init__(self, device):
        self.device = device
        super().__init__()
        self.character_embedding = CharEmbedding()
        self.contextual_embedding = nn.LSTM(input_size=100 + 300, hidden_size=100, bidirectional=True, batch_first=True, dropout=0.2)
        self.highway = Highway()
        self.attention_flow = AttentionFlow()
        self.modelling = nn.LSTM(input_size=800, hidden_size=100, bidirectional=True, batch_first=True, dropout=0.2, num_layers=2)
        self.output_start = nn.Linear(1000, 1, bias=False)
        self.modelling_end = nn.LSTM(input_size=200, hidden_size=100, bidirectional=True, batch_first=True, dropout=0.2)
        self.output_end = nn.Linear(1000, 1, bias=False)
        self.modelling_answerable = nn.LSTM(input_size=200, hidden_size=100, bidirectional=True, batch_first=True, dropout=0.2)
        self.output_answerable = nn.Linear(200, 1, bias=False)
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()
        self.loss_fn_index = nn.CrossEntropyLoss(reduction="none")
        self.loss_fn_answerable = nn.BCEWithLogitsLoss(reduction="none")
    def forward(self, x):
        x_contexts_char = x[0][0]
        x_contexts_glove = x[0][1]
        x_questions_char = x[1][0]
        x_questions_glove = x[1][1]

        x_contexts_char = self.character_embedding(x_contexts_char)
        x_questions_char = self.character_embedding(x_questions_char)

        x_contexts = torch.cat([x_contexts_char, x_contexts_glove], 2)
        x_questions = torch.cat([x_questions_char, x_questions_glove], 2)

        x_contexts = self.contextual_embedding(x_contexts)[0]
        x_questions = self.contextual_embedding(x_questions)[0]

        x_contexts = self.highway(x_contexts)
        x_questions = self.highway(x_questions)

        x = self.attention_flow([x_contexts, x_questions])
        x_start = self.modelling(x)[0]
        x_end = self.modelling_end(x_start)[0]
        x_answerable = self.modelling_answerable(x_end)[0]
        x_answerable = x_answerable.view(x.size()[0], x.size()[1], 2, 100)

        x_start = torch.cat([x_start, x], dim=-1)
        x_end = torch.cat([x_end, x], dim=-1)
        x_answerable = torch.cat([x_answerable[:, -1, 0], x_answerable[:, 0, 1]], dim=-1)

        start = self.output_start(x_start)
        end = self.output_end(x_end)
        start = torch.squeeze(start, dim=-1)
        end = torch.squeeze(end, dim=-1)
        answerable = self.output_answerable(x_answerable)
        return start, end, answerable
    def encode(self, text):
        tokens = self.tokenizer.tokenize(text)
        glove_embeddings = [get_word_embedding(token.lower()) for token in tokens]
        char_embeddings = [get_char_embeddings(token, index=True) for token in tokens]
        spans = list(self.tokenizer.span_tokenize(text))
        word_index = []
        for i in range(spans[0][0]):
            word_index.append(0)
        for i, span in enumerate(spans):
            if i + 1 < len(spans):
                num_of_chars = spans[i + 1][0] - span[0]
            else:
                num_of_chars = len(text) - span[0]
            for j in range(num_of_chars):
                word_index.append(i)
        return char_embeddings, np.array(glove_embeddings), (spans, word_index), text, tokens
    def dataset_embedding(self):
        def callback(context, question, answers, is_impossible, multiple_answers=False):
            context = context.encode("ascii", errors="replace").decode("ascii")
            question = question.encode("ascii", errors="replace").decode("ascii")
            x = (self.encode(context), self.encode(question))
            if multiple_answers:
                return x, (x[0][2][1], answers), (x[0][3], x[1][3]), (x[0][2][0], x[1][2][0]), False, 2
            if is_impossible:
                return x, (0, 0), (x[0][3], x[1][3]), (x[0][2][0], x[1][2][0]), True
            return x, (x[0][2][1][answers[0]["answer_start"]], x[0][2][1][answers[0]["answer_start"] + len(answers[0]["text"]) - 1]), (x[0][3], x[1][3]), (x[0][2][0], x[1][2][0]), False
        return callback
    def loss(self, x, y):
        return torch.mean((1 - y[2]) * (self.loss_fn_index(x[0], y[0]) + self.loss_fn_index(x[1], y[1])) + self.loss_fn_answerable(x[2], y[2]))

class CharEmbedding(nn.Module):
    def __init__(self, character_embedding_size=8, kernel_size=5, num_of_kernels=100):
        super().__init__()
        self.character_embedding_size = character_embedding_size
        self.num_of_kernels = num_of_kernels
        self.embedding = nn.Embedding(get_char_embedding_size(), character_embedding_size)
        self.conv = nn.Conv1d(character_embedding_size, num_of_kernels, kernel_size)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        batch_size = x.size()[0]
        num_of_words = x.size()[1]
        num_of_chars = x.size()[2]
        x = self.embedding(x)
        x = x.view(batch_size * num_of_words, num_of_chars, self.character_embedding_size)
        x = torch.transpose(x, 1, 2)
        x = self.conv(x)
        x = F.relu(x)
        x = F.max_pool1d(x, x.size()[2])
        x = x.view(batch_size, self.num_of_kernels, num_of_words)
        x = torch.transpose(x, 1, 2)
        return self.dropout(x)

class Highway(nn.Module):
    def __init__(self, num_of_layers=2, input_size=200):
        super().__init__()
        self.dense_layers = nn.ModuleList([nn.Linear(input_size, input_size) for i in range(num_of_layers)])
        self.gate_layers = nn.ModuleList([nn.Linear(input_size, input_size) for i in range(num_of_layers)])
    def forward(self, x):
        for dense_layer, gate_layer in zip(self.dense_layers, self.gate_layers):
            gate = torch.sigmoid(gate_layer(x))
            x = gate * F.relu(dense_layer(x)) + (1 - gate) * x
        return x

class AttentionFlow(nn.Module):
    def __init__(self, input_size=200):
        super().__init__()
        self.similarity = nn.Linear(input_size * 3, 1, bias=False)
    def forward(self, x):
        x_contexts, x_questions = x
        batch_size, num_of_context_words, embedding_size = x_contexts.size()
        num_of_question_words = x_questions.size()[1]
        shape = (batch_size, num_of_context_words, num_of_question_words, embedding_size)

        x_contexts_ = x_contexts.view(batch_size, num_of_context_words, 1, embedding_size)
        x_questions_ = x_questions.view(batch_size, 1, num_of_question_words, embedding_size)

        x_contexts_ = x_contexts_.expand(*shape)
        x_questions_ = x_questions_.expand(*shape)

        x = self.similarity(torch.cat([x_contexts_, x_questions_, x_contexts_ * x_questions_], -1))
        x = x.view(batch_size, num_of_context_words, num_of_question_words)

        context_to_question_attention = torch.bmm(F.softmax(x, dim=-1), x_questions)
        question_to_context_attention = torch.max(x, dim=-1)[0]
        question_to_context_attention = F.softmax(question_to_context_attention, dim=-1)
        question_to_context_attention = question_to_context_attention.view(batch_size, 1, num_of_context_words)
        question_to_context_attention = torch.bmm(question_to_context_attention, x_contexts)
        question_to_context_attention = question_to_context_attention.expand(batch_size, num_of_context_words, embedding_size)
        
        return torch.cat([x_contexts, context_to_question_attention, x_contexts * context_to_question_attention, x_contexts * question_to_context_attention], dim=-1)

def collate_fn(batch):
    context_char_shape = [len(batch), *np.maximum.reduce([[len(sentence[0][0][0]), *np.maximum.reduce([word.shape for word in sentence[0][0][0]])] for sentence in batch])]
    context_glove_shape = [len(batch), *np.maximum.reduce([[len(sentence[0][0][1]), *np.maximum.reduce([word.shape for word in sentence[0][0][1]])] for sentence in batch])]
    question_char_shape = [len(batch), *np.maximum.reduce([[len(sentence[0][1][0]), *np.maximum.reduce([word.shape for word in sentence[0][1][0]])] for sentence in batch])]
    question_glove_shape = [len(batch), *np.maximum.reduce([[len(sentence[0][1][1]), *np.maximum.reduce([word.shape for word in sentence[0][1][1]])] for sentence in batch])]

    x_contexts_char = np.zeros(context_char_shape)
    x_contexts_glove = np.zeros(context_glove_shape)
    x_questions_char = np.zeros(question_char_shape)
    x_questions_glove = np.zeros(question_glove_shape)

    for i, q_a_pair in enumerate(batch):
        for j, context_word_chars in enumerate(q_a_pair[0][0][0]):
            x_contexts_char[i, j, :context_word_chars.shape[0]] = context_word_chars
        x_contexts_glove[i, :q_a_pair[0][0][1].shape[0]] = q_a_pair[0][0][1]
        for j, question_word_chars in enumerate(q_a_pair[0][1][0]):
            x_questions_char[i, j, :question_word_chars.shape[0]] = question_word_chars
        x_questions_glove[i, :q_a_pair[0][1][1].shape[0]] = q_a_pair[0][1][1]

    if batch[0][-1] == 2:
        return ((torch.from_numpy(x_contexts_char).long(), torch.from_numpy(x_contexts_glove).float()), (torch.from_numpy(x_questions_char).long(), torch.from_numpy(x_questions_glove).float())), [q_a_pair[1] for q_a_pair in batch], ([q_a_pair[2][0] for q_a_pair in batch], [q_a_pair[2][1] for q_a_pair in batch]), ([q_a_pair[3][0] for q_a_pair in batch], [q_a_pair[3][1] for q_a_pair in batch])
    
    y_start = np.array([q_a_pair[1][0] for q_a_pair in batch])
    y_end = np.array([q_a_pair[1][1] for q_a_pair in batch])
    y_answerable = np.array([[float(q_a_pair[-1])] for q_a_pair in batch])

    return ((torch.from_numpy(x_contexts_char).long(), torch.from_numpy(x_contexts_glove).float()), (torch.from_numpy(x_questions_char).long(), torch.from_numpy(x_questions_glove).float())), (torch.from_numpy(y_start).long(), torch.from_numpy(y_end).long(), torch.from_numpy(y_answerable)), ([q_a_pair[2][0] for q_a_pair in batch], [q_a_pair[2][1] for q_a_pair in batch]), ([q_a_pair[3][0] for q_a_pair in batch], [q_a_pair[3][1] for q_a_pair in batch])

