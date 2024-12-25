from .model import Net
import nltk
from ..embeddings.glove import get_word_embedding
from ..embeddings.char import get_text_embeddings as get_char_embeddings
from ..embeddings.char import get_embedding_size as get_char_embedding_size
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
torch.autograd.set_detect_anomaly(True)

class QANet(Net):
    def __init__(self, device):
        self.device = device
        super().__init__()
        self.character_embedding = CharEmbedding()
        self.highway = Highway(input_size=400)
        self.embedding_projection = nn.Linear(400, 128)
        self.encoder = EncoderBlock(4)
        self.attention = Attention(input_size=128)
        self.model_encoder = nn.Sequential(*[EncoderBlock(2) for i in range(7)])
        self.output_start = nn.Linear(256, 1, bias=False)
        self.output_end = nn.Linear(256, 1, bias=False)
        self.output_answerable = nn.Linear(256, 1, bias=False)
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

        x_contexts = self.highway(x_contexts)
        x_questions = self.highway(x_questions)

        x_contexts = self.embedding_projection(x_contexts)
        x_questions = self.embedding_projection(x_questions)

        x_contexts = self.encoder(x_contexts)
        x_questions = self.encoder(x_questions)

        x = self.attention([x_contexts, x_questions])

        x = self.model_encoder(x)

        x_start = self.model_encoder(x)
        x_end = self.model_encoder(x)
        x_answerable = self.model_encoder(x)

        x_start = F.softmax(self.output_start(torch.cat([x, x_start], dim=-1)), dim=-1)
        x_end = F.softmax(self.output_end(torch.cat([x, x_end], dim=-1)), dim=-1)
        x_answerable = torch.sigmoid(self.output_answerable(torch.cat([x, x_answerable], dim=-1)))
        x_answerable = torch.mean(x_answerable, dim=-2)
        return x_start[:, :, 0], x_end[:, :, 0], torch.unsqueeze(x_answerable, -1)[:, 0]
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
        self.conv = DepthwiseSeparableConv1d(character_embedding_size, num_of_kernels, kernel_size)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        batch_size = x.size()[0]
        num_of_words = x.size()[1]
        num_of_chars = x.size()[2]
        x = self.embedding(x)
        x = x.view(batch_size * num_of_words, num_of_chars, self.character_embedding_size)
        x = self.conv(x)
        x = torch.transpose(x, 1, 2)
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

class EncoderBlock(nn.Module):
    def __init__(self, num_of_conv_layers):
        super().__init__()
        self.positional_encoding = PositionalEncoding(128)
        self.norm_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for i in range(num_of_conv_layers):
            self.norm_layers.append(nn.LayerNorm(128))
            self.conv_layers.append(DepthwiseSeparableConv1d(128, 128, 7))
        self.layer_norm_attention = nn.LayerNorm(128)
        self.multihead_attention = nn.MultiheadAttention(128, 8)
        self.layer_norm_feed_forward = nn.LayerNorm(128)
        self.feed_forward = nn.Linear(128, 128, bias=True)
    def forward(self, x):
        x = self.positional_encoding(x)
        for norm_layer, conv_layer in zip(self.norm_layers, self.conv_layers):
            x_ = norm_layer(x)
            x = x + conv_layer(x_)
        x_ = self.layer_norm_attention(x)
        x_ = torch.transpose(x_, 0, 1)
        x_ = self.multihead_attention(x_, x_, x_)
        x = x + torch.transpose(x_[0], 0, 1)
        x_ = self.layer_norm_feed_forward(x)
        x = x + F.relu(self.feed_forward(x_))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = np.zeros((max_len, d_model))
        position = np.expand_dims(np.arange(0, max_len, dtype=np.float32), 1)
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = np.expand_dims(pe, 0).transpose(1, 0, 2)
        pe = torch.from_numpy(pe).float()
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(input_dim, input_dim, kernel_size, padding=kernel_size // 2, groups=input_dim)
        self.pointwise_conv = nn.Conv1d(input_dim, output_dim, kernel_size=1, padding=0)

    def forward(self, x):
        y = torch.transpose(x, 1, 2)
        y = self.depthwise_conv(y)
        y = self.pointwise_conv(y)
        y = torch.transpose(y, 1, 2)
        return F.relu(y)

class Attention(nn.Module):
    def __init__(self, input_size=200):
        super().__init__()
        self.similarity = nn.Linear(input_size * 3, 1, bias=False)
        self.embedding_projection = nn.Linear(input_size * 4, 128)
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
        S_ = F.softmax(x, dim=-1)
        S__ = F.softmax(x, dim=-2)
        context_to_question_attention = torch.bmm(S_, x_questions)
        question_to_context_attention = torch.bmm(torch.bmm(S_, torch.transpose(S__, 1, 2)), x_contexts)
        return self.embedding_projection(torch.cat([x_contexts, context_to_question_attention, x_contexts * context_to_question_attention, x_contexts * question_to_context_attention], dim=-1))

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
