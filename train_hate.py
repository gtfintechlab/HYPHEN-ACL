import logging
import math
import pickle
import random
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

from model import HYPHEN
from preprocess_hate import CustomDataset
from radam import RiemannianAdam

print(torch.cuda.is_available())
parser = argparse.ArgumentParser(description="hyplstm model")

parser.add_argument(
    "--lr",
    default=5e-3,
    type=float,
    help="lr for training (default 5e-4)"
)

parser.add_argument(
    "--epochs",
    default=301,
    type=int,
    help="num of epochs (default 301)"
)

parser.add_argument(
    "--batch_size",
    default=128,
    type=int,
    help="num of epochs (default 301)"
)

parser.add_argument(
    "--attn_type",
    default="vanilla",
    type=str,
    help="attn type (default vanilla)"
)

parser.add_argument(
    "--lookback_days",
    default=30,
    type=int,
    help="lookback days (default 5)"
)

parser.add_argument(
    "--learnable_curvature",
    default=False,
    type=bool,
    help="learnable_curvature (default False)"
)

parser.add_argument(
    "--init_curvature_val",
    default=0.,
    type=float,
    help="init_curvature_val (default 0.)"
)

args = parser.parse_args()


SEED = 42
warnings.filterwarnings('ignore')
logging.basicConfig(
    filename=f'logs/train-logs-{args.attn_type}-attn-{args.lookback_days}-lookback-{args.epochs}-learnable_curvature-{args.learnable_curvature}-learn_init_val-{args.init_curvature_val}.txt',
    filemode='w',
    level=logging.INFO
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def train(model, criterion, speakers_train, epochs, optimizer, speakers_val, speakers_test, window=5, time=False, name_exp='hyp-hawkes'):
    best_val_acc = 0
    best_val_mcc = 0
    best_val_f1 = 0
    for epch in tqdm(range(epochs)):

        model.train()
        total_correct = []
        train_labels = []
        loss_train = 0.0
        total_train = 0
        for ds_train in speakers_train:
            optimizer.zero_grad()
            input_tensor = ds_train['speech_data']
            label = ds_train['label']
            label = label.long().squeeze(1)
            train_labels.extend(label.tolist())
            dates = ds_train['dates'].squeeze(1)
            dates_inv = ds_train['dates_inv'].squeeze(1)
            output = model(input_tensor.squeeze(
                1), dates, dates_inv).squeeze(1)
            pred = torch.argmax((output), -1).tolist()
            total_correct.extend(pred)
            loss = criterion(output, label)
            loss_train += loss*len(output)
            total_train += len(output)
            loss.backward()
            optimizer.step()
        tacc = accuracy_score(train_labels, total_correct)
        tf1 = f1_score(train_labels, total_correct, average='macro')
        if epch % 50 == 0:
            logger.info(
                "Train Accuracy at epoch {} is: {} and f1 is: {} and loss is {}".format(epch, tacc, tf1, loss_train/total_train))

        # Eval part
        model.eval()
        with torch.no_grad():
            total_correct_val = []
            val_labels = []
            loss_val = 0.0
            total_val = 0
            for ds_val in speakers_val:
                input_tensor_val = ds_val['speech_data']
                label_val = ds_val['label']
                label_val = label_val.long().squeeze(1)
                val_labels.extend(label_val.tolist())
                dates = ds_val['dates'].squeeze(1)
                dates_inv = ds_val['dates_inv'].squeeze(1)
                output = model(input_tensor_val.squeeze(1),
                               dates, dates_inv).squeeze(1)

                pred = torch.argmax((output), -1).tolist()
                total_correct_val.extend(pred)
                loss = criterion(output, label_val)
                loss_val += loss*len(output)
                total_val += len(output)

            tacc = accuracy_score(val_labels, total_correct_val)
            tf1 = f1_score(val_labels, total_correct_val, average='macro')
            if epch % 50 == 0:
                if tacc > best_val_acc:
                    best_val_acc = tacc
                if tf1 > best_val_f1:
                    torch.save(
                        {
                            "model_wts": model.state_dict(),
                            "current_epoch": epch,
                            "val_acc": tacc,
                            "val_f1": tf1,
                            "curvature_val": args.init_curvature_val,
                            "is_curvature": args.learnable_curvature,
                        },
                        "saved_models/"+"-epoch" +
                        str(epch)+"-"+name_exp +
                        f"lookback-{args.lookback_days}" +
                        f"learnable_curvature-{args.learnable_curvature}"+".pth",
                    )
                    logger.info(
                        "Val Accuracy at epoch {} is: {} and f1 is: {} and loss is {}".format(epch, tacc, tf1, loss_val/total_val))
                    logger.info(
                        f"Best Val mcc {best_val_mcc}"
                    )


if __name__ == "__main__":
    print("Attn_type", args.attn_type)
    print("Lookback_days", args.lookback_days)
    print("Learnable curvature", args.learnable_curvature)
    print("Data loaded")
    ds_train = CustomDataset(
        '/root/sanchit/research-group/hatespeech_history/df_train_hate.pkl', args.lookback_days)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size,
                          shuffle=True, drop_last=True)

    ds_val = CustomDataset(
        '/root/sanchit/research-group/hatespeech_history/df_val_hate.pkl', args.lookback_days)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size,
                        shuffle=True, drop_last=True)
    ds_test = CustomDataset(
        '/root/sanchit/research-group/hatespeech_history/df_test_hate.pkl', args.lookback_days)

    dl_test = DataLoader(ds_test, batch_size=args.batch_size,
                         shuffle=True, drop_last=True)

    criterion = nn.CrossEntropyLoss()
    model3 = HYPHEN(768, args.batch_size, args.batch_size,
                    attn_type=args.attn_type, learnable_curvature=args.learnable_curvature, init_curvature_val=args.init_curvature_val, n_class=4).to('cuda')
    for p in model3.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    optim_2 = RiemannianAdam(
        model3.parameters(), lr=args.lr, weight_decay=1e-4)
    train(model3, criterion, dl_train, args.epochs, optim_2,
          dl_val, dl_test, True, name_exp=args.attn_type)
