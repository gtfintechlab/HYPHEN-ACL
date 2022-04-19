from datetime import datetime

import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, dataset

PATH_SPEECH = "speech_numpys/"


class CustomDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.vals = []
        for i, data in self.df.iterrows():
            date = data["debate_id"][:10] #First 10 characters are the date here
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            label = int(data["vote"])
            speaker_id = int(data["speaker_id"])
            npy_embeddings = np.load(PATH_SPEECH + str(i) + ".npy")
            fin_dict = {
                "speech_data": torch.cuda.FloatTensor(npy_embeddings),
                "date_obj": date_obj,
                "label": label,
                "speaker_id": speaker_id,
            }
            self.vals.append(fin_dict)

    def __len__(self):
        return len(self.vals)

    def __getitem__(self, index):
        return self.vals[index]


class SpeakerDS(Dataset):
    def __init__(self, li):
        self.li = li

    def __len__(self):
        return len(self.li)

    def __getitem__(self, index):
        return self.li[index]


def prepare(df, window=5):
    speakers_test = {}
    ds = CustomDataset(df)
    for i in range(len(ds)):
        if ds[i]["speaker_id"] in speakers_test:
            speakers_test[ds[i]["speaker_id"]].append(
                {
                    "speech_data": ds[i]["speech_data"],
                    "date_obj": ds[i]["date_obj"],
                    "label": ds[i]["label"],
                }
        )
        else:
            speakers_test[ds[i]["speaker_id"]] = [
                {
                    "speech_data": ds[i]["speech_data"],
                    "date_obj": ds[i]["date_obj"],
                    "label": ds[i]["label"],
                }
            ]

    # window = 5
    skipped_vals = 0
    speaker_window_test_pkl = []
    for speaker, ds in speakers_test.items():
        if len(ds) < window:
            skipped_vals += len(ds)
        for idx in range(len(ds) - window):
            tmp_ds = ds[idx : idx + window]  # Sliding window
            li_speech_tensors = []
            dates = []
            label = ds[idx + window - 1]["label"]
            label = torch.FloatTensor([label]).to(
                "cuda"
            )  # last label to be checked with
            for data in tmp_ds:
                li_speech_tensors.append(data["speech_data"])
                dates.append(data["date_obj"])
            dates.sort()
            input_tensor = torch.cat(li_speech_tensors, dim=0)  # (window X 768)
            input_tensor = input_tensor.unsqueeze(0).to("cuda")
            diff = [0] * len(dates)
            tmp = diff
            diff[0] = 1
            for i in range(1, len(dates)):
                diff[i] = int(
                    (dates[i] - dates[i - 1]).total_seconds() // (window * 86400)
                )
                tmp[i] = diff[i]
                if diff[i] == 0:
                    diff[i] = 1
                else:
                    diff[i] = 1 / diff[i]
            diff = torch.FloatTensor([diff]).to("cuda")
            tmp = torch.FloatTensor([tmp]).to("cuda")
            tmp_dict = {
                "speech_data": input_tensor,
                "label": label,
                "dates_inv": diff,
                "dates": tmp,
            }
            speaker_window_test_pkl.append(tmp_dict)
    return speaker_window_test_pkl
