import argparse
from ast import Num
import copy
import logging
import pickle
import sys
import warnings
import json
from collections import Counter
import numpy as np
from transformers import AdamW, get_cosine_schedule_with_warmup
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from earlystopping import EarlyStopping
import yaml
import math
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_suicide import SuicidalDataset
from model_hyphen import HYPHEN
from radam import RiemannianAdam

torch.autograd.set_detect_anomaly(True)
print(torch.cuda.is_available())
with open("parameters_suicide.yaml", "r") as f:
    args = yaml.load(f, Loader=yaml.FullLoader)


SEED = 42
warnings.filterwarnings("ignore")
logging.basicConfig(
    filename=f"logs/train-logs-{args['attn_type']}-attn-{args['lookback_days']}-lookback-{args['epoch']}-learnable_curvature-{args['learnable_curvature']}-learn_init_val-{args['init_curvature_value']}-hyp-lstm-{args['hyp_lstm']}--diff_timestamp_without_sampler.txt",
    filemode="w",
    level=logging.INFO,
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.log(logging.INFO, args)


def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = F.binary_cross_entropy_with_logits(
        input=logits, target=labels, reduction="none"
    )

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(
            -gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits))
        )

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights, dtype=torch.float32).cuda()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(
            input=logits, target=labels_one_hot, weight=weights
        )
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        cb_loss = F.binary_cross_entropy(
            input=pred, target=labels_one_hot, weight=weights
        )

    return cb_loss


def get_mcc_score(labels, pred):
    confus_matrix = confusion_matrix(labels, pred)
    tn, fp, fn, tp = confus_matrix.ravel()
    tacc = (tp + tn) / (tp + tn + fp + fn)
    tf1 = (tp) / (tp + 0.5 * (fp + fn))
    tmcc = (tp * tn - fp * fn) / (
        math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    return tmcc


def train(model, optimizer, loss_fn, train_loader, dataset_len):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for bi, inputs in enumerate(tqdm(train_loader)):
        labels = inputs["label"]
        inp_tensor = inputs["temporal_tweet_features"]
        normal_timestamp = inputs["normal_timestamp"]
        inverse_timestamp = inputs["inverse_timestamp"]

        optimizer.zero_grad()
        output = model(inp_tensor, normal_timestamp, inverse_timestamp)
        pred = torch.argmax((output), -1)
        loss = loss_fn(output, labels, labels.unique(return_counts=True)[1].tolist())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_corrects += torch.sum(pred == labels).item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = running_corrects / (dataset_len)
    return epoch_loss, epoch_acc


def eval_loop(model, dataloader, device, dataset_len):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    fin_targets = []
    fin_outputs = []

    for bi, inputs in enumerate(tqdm(dataloader, total=len(dataloader), leave=False)):
        labels = inputs["label"]
        inp_tensor = inputs["temporal_tweet_features"]
        normal_timestamp = inputs["normal_timestamp"]
        inverse_timestamp = inputs["inverse_timestamp"]

        with torch.no_grad():
            output = model(inp_tensor, normal_timestamp, inverse_timestamp)

        _, preds = torch.max(output, 1)
        loss = loss_fn(output, labels, labels.unique(return_counts=True)[1].tolist())
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

        fin_targets.append(labels.cpu().detach().numpy())
        fin_outputs.append(preds.cpu().detach().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_corrects.double() / dataset_len

    return epoch_loss, epoch_accuracy, np.hstack(fin_outputs), np.hstack(fin_targets)


def loss_fn(output, targets, samples_per_cls):
    beta = 0.9999
    gamma = 2.0
    no_of_classes = 2
    loss_type = "focal"

    return CB_loss(
        targets, output, samples_per_cls, no_of_classes, loss_type, beta, gamma
    )


def main(args=None):
    EPOCHS = args["epoch"]
    BATCH_SIZE = args["batch_size"]
    HIDDEN_DIM = args["hidden_dim"]

    with open("data/train_1_enc.pkl", "rb") as f:
        train_1 = pickle.load(f)
    with open("data/train_2.pkl", "rb") as f:
        train_2 = pickle.load(f)
    with open("data/train_3.pkl", "rb") as f:
        train_3 = pickle.load(f)
    with open("data/val_enc.pkl", "rb") as f:
        df_val = pickle.load(f)
    df_train = pd.concat([train_1, train_2, train_3], ignore_index=True)
    df_train_label_counts = df_train["label"].value_counts().to_dict()
    df_label_counts = [df_train_label_counts[k] for k in df_train_label_counts]
    weights = sum(df_label_counts) / np.array(df_label_counts, dtype="float")
    labels = df_train["label"].to_list()
    sample_weights = [weights[i] for i in labels]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        sample_weights, len(sample_weights)
    )
    print("Data loaded train")
    print("Len df", len(df_train))
    ds_train = SuicidalDataset(
        df_train.label.values,
        df_train.enc.values,
        df_train.hist_dates.values,
        window=args["lookback_days"],
        unit=args["unit"],
    )
    ds_val = SuicidalDataset(
        df_val.label.values,
        df_val.enc.values,
        df_val.hist_dates.values,
        window=args["lookback_days"],
        unit=args["unit"],
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        drop_last=True,
    )
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = HYPHEN(
        768,
        HIDDEN_DIM,
        BATCH_SIZE,
        attn_type=args["attn_type"],
        learnable_curvature=args["learnable_curvature"],
        init_curvature_val=args["init_curvature_value"],
        time_param=args["time_param"],
    ).to(device)
    best_mcc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    optim = AdamW(model.parameters(), lr=args["lr"])
    early_stopping = EarlyStopping(patience=args["patience"], verbose=True)
    for epoch in tqdm(range(EPOCHS)):
        loss, accuracy = train(model, optim, loss_fn, dl_train, len(ds_train))
        eval_loss, eval_accuracy, pred, targets = eval_loop(
            model, dl_val, device, len(ds_val)
        )
        metric = f1_score(targets, pred, average="macro")
        mcc = get_mcc_score(targets, pred)
        recall_1 = recall_score(targets, pred, average=None)[1]
        early_stopping(eval_loss, model, mcc)
        if mcc > best_mcc:
            best_mcc = mcc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "best_model_wts": best_model_wts,
                    "BATCH_SIZE": BATCH_SIZE,
                    "HIDDEN_DIM": HIDDEN_DIM,
                    "best_metric": best_mcc,
                    "epoch": epoch,
                },
                f"saved_models/{args['attn_type']}_{args['lookback_days']}_{best_mcc}_hyp_lstm-{args['hyp_lstm']}_learnable_curvature-{args['learnable_curvature']}_init_curvature_value-{args['init_curvature_value']}_unit-{args['unit']}"
                + ".pt",
            )
            result = classification_report(
                targets, pred, labels=[0, 1], output_dict=True
            )
            with open(
                f"results_suicide/{args['attn_type']}_{args['lookback_days']}_{best_mcc}_hyp_lstm-{args['hyp_lstm']}_learnable_curvature-{args['learnable_curvature']}"
                + ".json",
                "w",
            ) as f:
                json.dump(result, f)
        logger.info(
            "Train Accuracy at epoch {} is: {} and loss is {}".format(
                epoch, accuracy, loss
            )
        )
        logger.info(
            f"Val accuracy is {eval_accuracy} and loss is {eval_loss} and macro-f1 is {metric} and mcc is {mcc} and recall_1 is {recall_1}"
        )
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break


if __name__ == "__main__":
    main(args)
