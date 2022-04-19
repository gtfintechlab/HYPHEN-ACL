import pickle

import torch
from torch.utils.data import Dataset
import numpy as np
import datetime


class SuicidalDataset(Dataset):
    def __init__(
        self,
        label,
        temporal,
        timestamp,
        window=5,
        unit=30,
    ):
        super().__init__()
        self.label = label
        self.window = window
        self.temporal = temporal
        self.timestamp = timestamp
        self.unit = unit

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        labels = torch.tensor(self.label[item], device="cuda")
        if len(self.temporal[item]) < self.window:
            temporal_tweet_features = torch.zeros(
                (self.window, 768), dtype=torch.float32, device="cuda"
            )
            normal_timestamp = torch.zeros(
                (self.window), dtype=torch.float32, device="cuda"
            )
            inverse_timestamp = torch.ones(
                (self.window), dtype=torch.float32, device="cuda"
            )
        else:
            temporal_tweet_features = torch.tensor(
                self.temporal[item][: self.window], dtype=torch.float32, device="cuda"
            )
            timestamps = self.timestamp[item][: self.window]
            # the timestamps are in decreasing order. So last 5 implies first 5
            normal_timestamp, inverse_timestamp = get_timestamp(timestamps, self.unit)
        return {
            "label": labels,
            "temporal_tweet_features": temporal_tweet_features,
            "normal_timestamp": torch.tensor(
                normal_timestamp, dtype=torch.float32, device="cuda"
            ),
            "inverse_timestamp": torch.tensor(
                inverse_timestamp, dtype=torch.float32, device="cuda"
            ),
        }


def get_timestamp(dates, unit = 30):
    normal_dates = [0] * len(dates)
    normal_dates[0] = 1
    for i in range(1, len(dates)):
        normal_dates[i] = ((dates[i - 1] - dates[i]).total_seconds()) // (86400 * unit)
    inv_dates = [1 / val if val != 0 else 1 for val in normal_dates]
    return normal_dates, inv_dates


# def get_timestamp(x):
#     timestamp = []
#     for t in x:
#         timestamp.append(datetime.datetime.timestamp(t))

#     (np.array(timestamp) - timestamp[-1])
#     timestamp = [t / (86400 * 30) for t in timestamp]
#     inv_dates = [1 / val if val != 0 else 1 for val in timestamp]
#     return timestamp, inv_dates


if __name__ == "__main__":
    with open("/root/sanchit/research-group/data/train_1_enc.pkl", "rb") as f:
        df = pickle.load(f)
    dates = df["hist_dates"].values
    # ds = SuicidalDataset(
    #     df["label"].values,
    #     df["enc"].values,
    #     df["hist_dates"].values,
    #     window=5,
    # )
    # print(ds[0])
    normal_dates = []
    # for dt in dates:
    #     print(dt)
    #     time,inv = get_timestamp(dt)
    #     normal_dates.append(time)

    # std = [np.std(dt) for dt in normal_dates]
    # std = sorted(std, reverse=True)
    # mean = [np.mean(dt) for dt in normal_dates]
    # mean = sorted(mean, reverse=True)
    # print(mean[:10])
    # print(std[:10])
    # print(mean,'mean')
