import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, dataset
import torch
from collections import defaultdict


class CustomDataset(Dataset):
    def __init__(self, data_path, window=30):
        self.df = pd.read_pickle(data_path)
        self.data = self.prepare(window)

    def prepare(self, window):
        fin_data = []
        for _, data in self.df.iterrows():
            if len(data['enc']) < window or len(data['hist_dates']) < window:
                add_rows = 0
                if len(data['enc']) < window:
                    add_rows = window - len(data['enc'])
                if len(data['hist_dates']) < window:
                    add_rows = window - len(data['hist_dates'])
                if len(data['enc']) < window and len(data['hist_dates']) < window:
                    add_rows = window - \
                        max(len(data['enc']), len(data['hist_dates']))
                if add_rows:
                    row_stack = np.zeros((add_rows, 768))
                    data['enc'] = np.vstack((data['enc'], row_stack))
                    date = pd.Timestamp('1970-01-01')
                    date_arr = [date] * add_rows
                    data['hist_dates'].extend(date_arr)
            date_enc_dic = defaultdict(list)
            for i in range(len(data['hist_dates'])):
                date_enc_dic[data['hist_dates'][i]].append(data['enc'][i])
            date_enc_dic = {k: date_enc_dic[k] for k in sorted(date_enc_dic)}
            sorted_dates = []
            sorted_enc = []
            for key, val in date_enc_dic.items():
                tmp = [key]*len(val)
                sorted_dates.extend(tmp)
                sorted_enc.extend(val)
            for idx in range(len(date_enc_dic) - window+1):
                tmp_ds = sorted_enc[idx:idx+window]
                tmp_ds = np.array(tmp_ds)
                input_tensor = torch.FloatTensor(tmp_ds).to('cuda')
                label = data['label']
                label = torch.FloatTensor([label]).to(
                    'cuda')  # last label to be checked with
                dates = sorted_dates[idx:idx+window]
                input_tensor = input_tensor.unsqueeze(0).to('cuda')
                diff = [0]*len(dates)
                tmp = diff.copy()
                diff[0] = 1
                for i in range(1, len(dates)):
                    diff[i] = (
                        int((dates[i]-dates[i-1]).total_seconds()//86400))
                    if diff[i] == 0:
                        diff[i] = 1
                    else:
                        diff[i] = 1 / diff[i]
                tmp = torch.FloatTensor([tmp]).to('cuda')
                diff = torch.FloatTensor([diff]).to('cuda')
                tmp_dict = {
                    "speech_data": input_tensor,
                    "label": label,
                    "dates_inv": diff,
                    "dates": tmp
                }
                fin_data.append(tmp_dict)
        return fin_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    data_path = "hatespeech_history/df_test_hate.pkl"
    window = 10
    dataset = CustomDataset(data_path, window)
