import logging
import math
import pickle
import random
import sys
import warnings

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from model_hyphen import HYPHEN
from preprocess_data import SpeakerDS, prepare
from radam import RiemannianAdam

print(torch.cuda.is_available())
with open("parameters_parlvote.yaml", "r") as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

SEED = 42
warnings.filterwarnings("ignore")
logging.basicConfig(
    filename=f"logs/train-logs-{args['attn_type']}-attn-{args['lookback_days']}-lookback-{args['epochs']}-learnable_curvature-{args['learnable_curvature']}-learn_init_val-{args['init_curvature_value']}.txt",
    filemode="w",
    level=logging.INFO,
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def train(
    model,
    criterion,
    speakers_train,
    epochs,
    optimizer,
    speakers_val,
    window=5,
    time=False,
    name_exp="hyp-hawkes",
):
    best_val_acc = 0
    best_val_mcc = 0
    for epch in tqdm(range(epochs)):

        model.train()
        total_correct = []
        train_labels = []
        loss_train = 0.0
        total_train = 0
        for ds_train in speakers_train:
            optimizer.zero_grad()
            input_tensor = ds_train["speech_data"]
            label = ds_train["label"]
            label = label.long().squeeze(1)
            train_labels.extend(label.tolist())
            dates = ds_train["dates"].squeeze(1)
            dates_inv = ds_train["dates_inv"].squeeze(1)
            output = model(input_tensor.squeeze(1), dates, dates_inv).squeeze(1)
            pred = torch.argmax((output), -1).tolist()
            total_correct.extend(pred)
            loss = criterion(output, label)
            loss_train += loss * len(output)
            total_train += len(output)
            loss.backward()
            optimizer.step()
        confus_matrix = confusion_matrix(train_labels, total_correct)
        tn, fp, fn, tp = confus_matrix.ravel()
        tacc = (tp + tn) / (tp + tn + fp + fn)
        tf1 = (tp) / (tp + 0.5 * (fp + fn))
        tmcc = (tp * tn - fp * fn) / (
            math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        )
        if epch % 50 == 0:
            logger.info(
                "Train Accuracy at epoch {} is: {} and mcc is: {} and f1 is: {} and loss is {}".format(
                    epch, tacc, tmcc, tf1, loss_train / total_train
                )
            )

        # Eval part
        model.eval()
        with torch.no_grad():
            total_correct_val = []
            val_labels = []
            loss_val = 0.0
            total_val = 0
            for ds_val in speakers_val:
                input_tensor_val = ds_val["speech_data"]
                label_val = ds_val["label"]
                label_val = label_val.long().squeeze(1)
                val_labels.extend(label_val.tolist())
                dates = ds_val["dates"].squeeze(1)
                dates_inv = ds_val["dates_inv"].squeeze(1)
                output = model(input_tensor_val.squeeze(1), dates, dates_inv).squeeze(1)

                pred = torch.argmax((output), -1).tolist()
                total_correct_val.extend(pred)
                loss = criterion(output, label_val)
                loss_val += loss * len(output)
                total_val += len(output)

            confus_matrix = confusion_matrix(val_labels, total_correct_val)
            tn, fp, fn, tp = confus_matrix.ravel()
            tacc = (tp + tn) / (tp + tn + fp + fn)
            tf1 = (tp) / (tp + 0.5 * (fp + fn))
            tmcc = (tp * tn - fp * fn) / (
                math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            )
            if epch % 50 == 0:
                if tacc > best_val_acc:
                    best_val_acc = tacc
                if tmcc > best_val_mcc:
                    best_val_mcc = tmcc
                    torch.save(
                        {
                            "model_wts": model.state_dict(),
                            "current_epoch": epch,
                            "mcc": tmcc,
                            "val_acc": tacc,
                            "curvature_val": args["init_curvature_value"],
                            "is_curvature": args["learnable_curvature"],
                        },
                        "saved_models/"
                        + "-epoch"
                        + str(epch)
                        + "-"
                        + name_exp
                        + f"lookback-{args['lookback_days']}"
                        + f"learnable_curvature-{args['learnable_curvature']}"
                        + ".pth",
                    )
                    logger.info(
                        "Val Accuracy at epoch {} is: {} and mcc is: {} and f1 is: {} and loss is {}".format(
                            epch, tacc, tmcc, tf1, loss_val / total_val
                        )
                    )
                    logger.info(f"Best Val mcc {best_val_mcc}")


def main():
    EPOCHS = args["epoch"]
    BATCH_SIZE = args["batch_size"]
    HIDDEN_DIM = args["hidden_dim"]
    df = pd.read_csv("ParlVote_concat.csv")
    print("Attn_type", args["attn_type"])
    print("Lookback_days", args["lookback_days"])
    print("Learnable curvature", args["learnable_curvature"])
    train_num = int(0.7 * len(df))
    val_num = int(0.15 * len(df))
    train_df = df.iloc[:train_num]

    val_df = df.iloc[train_num + 1 : train_num + val_num]

    test_df = df.iloc[train_num + val_num + 1 :]

    speakers_train = prepare(train_df, window=args["lookback_days"])
    speakers_val = prepare(val_df, window=args["lookback_days"])
    speakers_test = prepare(test_df, window=args["lookback_days"])
    print("Data loaded")

    ds_train = SpeakerDS(speakers_train)
    dl_train = DataLoader(
        ds_train, batch_size=args["batch_size"], shuffle=True, drop_last=True
    )

    random.Random(SEED).shuffle(speakers_test)
    random.Random(SEED).shuffle(speakers_val)

    ds_val = SpeakerDS(speakers_val)
    dl_val = DataLoader(
        ds_val, batch_size=args["batch_size"], shuffle=True, drop_last=True
    )

    criterion = nn.CrossEntropyLoss()

    model = HYPHEN(
        768,
        HIDDEN_DIM,
        BATCH_SIZE,
        attn_type=args["attn_type"],
        learnable_curvature=args["learnable_curvature"],
        init_curvature_val=args["init_curvature_value"],
        time_param=args["time_param"],
    ).to("cuda")

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    optim_2 = RiemannianAdam(model.parameters(), lr=args["lr"], weight_decay=1e-4)

    train(
        model,
        criterion,
        dl_train,
        EPOCHS,
        optim_2,
        dl_val,
        True,
        name_exp=args["attn_type"],
    )


if __name__ == "__main__":
    main()
