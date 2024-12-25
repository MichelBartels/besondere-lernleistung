import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import wandb # API-Key muss als Umgebungsvariable übergeben werden
import numpy as np
import os
from tqdm import tqdm
import torch.nn.functional as F

# Aus dem offiziellen Evaluierungsskript entnommen
from collections import Counter
import string
import re

def normalize_answer(s):
    """Convert to lowercase and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_em(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

class Net(nn.Module):
    def train_model(self, dataset, test_dataset, eval_dataset, optimizer, learning_rate, loss, num_of_steps, batch_size, checkpoint_folder, log_steps=1000, test_steps=20000, weight_decay=0):
        os.makedirs(checkpoint_folder, exist_ok=True)
        wandb.init()
        if isinstance(loss, str):
            loss = loss.lower()
        optimizer = optimizer.lower()

        if loss == "cross_entropy":
            criterion = nn.CrossEntropyLoss()
        elif loss == "binary_cross_entropy":
            criterion = nn.BCELoss()
        elif loss == "mse" or loss == "l2":
            criterion = nn.MSELoss()
        elif loss == "l1":
            criterion = nn.L1Loss()
        else:
            criterion = loss
        
        if optimizer == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "adam":
            optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "adadelta":
            optimizer = optim.Adadelta(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "adagrad":
            optimizer = optim.Adagrad(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == "rmsprop":
            optimizer = optim.RMSprop(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        dataloader = data.DataLoader(dataset, batch_size=batch_size, num_workers=4, collate_fn=dataset.collate_fn)
        test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, num_workers=1, collate_fn=dataset.collate_fn)
        test_data_iterator = iter(test_dataloader)

        running_loss = 0
        for (x, y, _, _), step in tqdm(zip(dataloader, range(num_of_steps)), total=num_of_steps):
            if self.device:
                x = [[x__.to(self.device) for x__ in x_] for x_ in x]
                y = [y_.to(self.device) for y_ in y]
            self.zero_grad()
            outputs = self.__call__(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (step + 1) % log_steps == 0:
                self.eval()
                print("Step: " + str(step + 1) + " Loss: " + str(running_loss / log_steps))
                running_loss = 0
                table = wandb.Table(columns=["Context", "Question", "Right answer", "Predicted answer"])
                x, y, text, slides = next(test_data_iterator)
                if self.device:
                    x = [[x__.to(self.device) for x__ in x_] for x_ in x]
                    y = [y_.to(self.device) for y_ in y]
                outputs = self.__call__(x)

                y = [y_.cpu().detach().numpy() for y_ in y]
                outputs = [F.softmax(outputs_, dim=-1).cpu().detach().numpy() for outputs_ in outputs[:2]] + [torch.sigmoid(outputs[-1]).cpu().detach().numpy()]
                for correct_start_index, correct_end_index, correct_is_impossible, predicted_start_index, predicted_end_index, predicted_is_impossible, context, question, context_slide, question_slide in zip(*y, *outputs, *text, *slides):

                    predicted_start_index = predicted_start_index[:len(context_slide)]
                    predicted_end_index = predicted_end_index[:len(context_slide)]
                    start = -1
                    end = -1
                    start_end_probability = -1
                    for start_index, start_probability in enumerate(predicted_start_index):
                        if start_probability < start_end_probability:
                            continue
                        for end_index, end_probability in enumerate(predicted_end_index[start_index + 1:]):
                            if end_probability * start_probability < start_end_probability:
                                continue
                            start_end_probability = start_probability * end_probability
                            start = start_index
                            end = start_index + 1 + end_index
                    start = context_slide[start][0]
                    end = context_slide[end][1]
                    predicted_answer = context[start:end]
                    correct_start_index = context_slide[correct_start_index][0]
                    correct_end_index = context_slide[correct_end_index][1]
                    correct_answer = context[correct_start_index:correct_end_index]
                    if predicted_is_impossible >= 0.5:
                        predicted_answer = ""
                    if correct_is_impossible == 1:
                        correct_answer = ""
                    table.add_data(context, question, correct_answer, predicted_answer)
                wandb.log({"Examples": table}, step=step + 1)
                self.train()

            if (step + 1) % test_steps == 0:
                print("Calculating F1 and EM scores on test set")
                true_positives = 0
                false_positives = 0
                false_negatives = 0
                exact_matches = 0
                not_exact_matches = 0
                self.eval()
                with torch.no_grad():
                    total_f1 = 0
                    total_em = 0
                    total = 0
                    for x, y, text, slides in data.DataLoader(eval_dataset, batch_size=batch_size, num_workers=1, collate_fn=dataset.collate_fn):
                        if self.device:
                            x = [[x__.to(self.device) for x__ in x_] for x_ in x]
                        outputs = self.__call__(x)

                        outputs = [F.softmax(outputs_, dim=-1).cpu().detach().numpy() for outputs_ in outputs[:2]] + [torch.sigmoid(outputs[-1]).cpu().detach().numpy()]
                        for (slides, answers), predicted_start_index, predicted_end_index, predicted_is_impossible, context, question, context_slide, question_slide in zip(y, *outputs, *text, *slides):
                            if len(answers) == 0:
                                if predicted_is_impossible >= 0.5:
                                    total_em += 1
                                    total_f1 += 1
                                total += 1
                                continue
                            if predicted_is_impossible >= 0.5:
                                total_f1 += 1
                                continue
                            predicted_start_index = predicted_start_index[:len(context_slide)]
                            predicted_end_index = predicted_end_index[:len(context_slide)]
                            start = -1
                            end = -1
                            start_end_probability = -1
                            for start_index, start_probability in enumerate(predicted_start_index):
                                if start_probability < start_end_probability:
                                    continue
                                for end_index, end_probability in enumerate(predicted_end_index[start_index + 1:]):
                                    if end_probability * start_probability < start_end_probability:
                                        continue
                                    start_end_probability = start_probability * end_probability
                                    start = start_index
                                    end = start_index + 1 + end_index
                            start = context_slide[start][0]
                            end = context_slide[end][1]
                            predicted_answer = context[start:end]

                            f1 = []
                            em = []
                            for answer in answers:
                                f1.append(compute_f1(answer["text"], predicted_answer))
                                em.append(compute_em(answer["text"], predicted_answer))
                            total_f1 += max(f1)
                            total_em += max(em)
                            total += 1
                    f1 = total_f1 / total
                    em = total_em / total
                    wandb.log({"F1": f1 * 100, "EM": em * 100}, step=step + 1)
                    torch.save(self.state_dict(), os.path.join(checkpoint_folder, str(step + 1) + "_state_dict.pt"))
                    self.train()

            wandb.log({"Loss": loss.cpu().detach().numpy()}, step=step + 1)
        print("Finished training")
