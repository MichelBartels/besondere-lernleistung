from data_loader import SQuAD
import torch
from torch import nn
import torch.nn.functional as F
import wandb
from torchtext.data import Iterator
from tqdm import tqdm, trange
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve
#from torch.cuda.amp import autocast, GradScaler
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup

wandb.init(project="search")
wandb.config.batch_size = 4
wandb.config.learning_rate = 0.00005
wandb.config.mixed_precision = False
wandb.config.epochs = 4
device = torch.device("cuda:0")

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert.train()
    def forward(self, x):
        x = self.bert(**x)[0]
        x = torch.mean(x, 1)
        x = F.normalize(x, p=2, dim=1)
        return x

class Comparer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                nn.Linear(768 * 2, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
        )

    def forward(self, paragraph_encoding, question_encoding):
        x = torch.cat([paragraph_encoding, question_encoding], dim=1)
        x = self.model(x)
        return x

def dict_to_device(dict_, device):
    return {k: v.to(device) for k, v in dict_.items()}

def main():
    train_dataset = SQuAD(wandb.config.batch_size)
    test_dataset = SQuAD(wandb.config.batch_size, test=True)
    encoder = Encoder().to(device)
    comparer = Comparer().to(device)
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    parameters = list(encoder.parameters()) + list(comparer.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=wandb.config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=wandb.config.epochs * len(train_dataset))
    #scaler = GradScaler()
    train_iterator = iter(train_dataset)
    test_iterator = iter(test_dataset)
    step = 0
    for epoch in trange(wandb.config.epochs):
        for paragraph, positive, negative in tqdm(train_iterator):
            optimizer.zero_grad()
            paragraph = dict_to_device(paragraph, device)
            positive = dict_to_device(positive, device)
            negative = dict_to_device(negative, device)
            #with autocast(enabled=wandb.config.mixed_precision):
            if True:
                paragraph_encoding = encoder(paragraph)
                positive_encoding = encoder(positive)
                negative_encoding = encoder(negative)
                positive_results = comparer(paragraph_encoding, positive_encoding)
                negative_results = comparer(paragraph_encoding, negative_encoding)
                loss = loss_fn(positive_results, torch.zeros_like(positive_results)) + loss_fn(negative_results, torch.ones_like(negative_results))
            if wandb.config.mixed_precision:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1)
                optimizer.step()
            scheduler.step()
            if step % 4 == 0:
                wandb.log({"loss": loss}, commit=False, step=step)
            step += 1
        with torch.no_grad():
            paragraph_encoder.eval()
            question_encoder.eval()
            comparer.eval()
            positive_results_ = np.array([])
            negative_results_ = np.array([])
            for batch in tqdm(test_iterator):
                paragraph = batch.paragraph
                positive = batch.positive
                negative = batch.negative
                paragraph_encoding = paragraph_encoder(paragraph)
                positive_encoding = question_encoder(positive)
                negative_encoding = question_encoder(negative)
                positive_results = torch.sigmoid(comparer(paragraph_encoding, positive_encoding))
                negative_results = torch.sigmoid(comparer(paragraph_encoding, negative_encoding))
                positive_results_ = np.concatenate([positive_results_, positive_results.flatten().cpu().numpy()])
                negative_results_ = np.concatenate([negative_results_, negative_results.flatten().cpu().numpy()])
            results = np.concatenate([positive_results_, negative_results_])
            real_results = np.concatenate([np.zeros_like(positive_results_), np.ones_like(negative_results_)])
            cm = confusion_matrix(real_results, (results >= 0.5).astype(int), normalize="pred")
            roc = roc_curve(real_results, results)
            roc = map(lambda x: [0, x[0], x[1]], zip(*roc))
            roc = list(roc)[0:wandb.Table.MAX_ROWS]
            class_labels = ["positive", "negative"]
            wandb.log({"accuracy": (np.count_nonzero(positive_results_ < 0.5) + np.count_nonzero(negative_results_ >= 0.5)) / (2 * positive_results_.shape[0]), "true negatives": cm[0, 0], "false positives": cm[0, 1], "false negatives": cm[1, 0], "true positives": cm[1, 1]}, commit=False, step=step)
            """), "confusion matrix": wandb.plots.HeatMap(class_labels, class_labels, matrix_values=cm, show_text=True), "roc": wandb.visualize("wandb/roc/v1", wandb.Table(columns=["class", "fpr", "tpr"], data=roc))"""
            paragraph_encoder.train()
            question_encoder.train()
            comparer.train()

if __name__ == "__main__":
    main()
