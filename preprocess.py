from datetime import datetime

import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, dataset

PATH_SPEECH = 'speech_numpys/'


class CustomDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.vals = []
        for i, data in self.df.iterrows():
            date = data['debate_id'][:10]
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            label = int(data['vote'])
            speaker_id = int(data['speaker_id'])
            npy_embeddings = np.load(PATH_SPEECH + str(i) + '.npy')
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
    
    def __getitem__(self,index):
        return self.li[index]


def prepare(df, window = 5):
    speakers_test = {} 
    ds = CustomDataset(df)
    for i in range(len(ds)):
        if ds[i]['speaker_id'] in speakers_test:
            speakers_test[ds[i]['speaker_id']].append({'speech_data': ds[i]['speech_data'], 'date_obj': ds[i]['date_obj'], 'label': ds[i]['label']})
        else:
            speakers_test[ds[i]['speaker_id']] = [{'speech_data': ds[i]['speech_data'], 'date_obj': ds[i]['date_obj'], 'label': ds[i]['label']}]
    

    # window = 5
    skipped_vals = 0
    speaker_window_test_pkl = []
    deviations = []
    deviations2 = []
    for speaker, ds in speakers_test.items():
        # print(len(ds))
        if len(ds) < window:
            skipped_vals += len(ds)
        for idx in range(len(ds) - window):
                tmp_ds = ds[idx:idx+window]  # Sliding window
                li_speech_tensors = []
                dates = []
                label = ds[idx+window - 1]['label']
                label = torch.FloatTensor([label]).to(
                    'cuda')  # last label to be checked with
                for data in tmp_ds:
                    li_speech_tensors.append(data['speech_data'])
                    dates.append(data['date_obj'])
                dates.sort()
                input_tensor = torch.cat(
                    li_speech_tensors, dim=0)  # (window X 768)
                input_tensor = input_tensor.unsqueeze(0).to('cuda')
                # print(dates,'dates_hereee')
                diff = [0]*len(dates)
                tmp = diff
                diff[0] = 1
                for i in range(1,len(dates)):
                    diff[i] = (int((dates[i]-dates[i-1]).total_seconds()//5184000))
                    tmp[i] = diff[i]
                    if diff[i] == 0:
                        diff[i] = 1
                    else:
                        diff[i] = 1 / diff[i]
                # deviations.append(np.std(diff))
                # deviations2.append(np.std(tmp))
                # print(np.std(diff),'diff_heree')
                diff = torch.FloatTensor([diff]).to('cuda')
                tmp = torch.FloatTensor([tmp]).to('cuda')
                tmp_dict ={
                    "speech_data": input_tensor,
                    "label": label,
                    "dates_inv": diff,
                    "dates": tmp
                }
                speaker_window_test_pkl.append(tmp_dict)
    # print(min(deviations), max(deviations), "dev")
    # print(min(deviations2), max(deviations2), "dev2")
    return speaker_window_test_pkl
    

if __name__ == "__main__":
    df = pd.read_csv('ParlVote_concat.csv')
    # df = df[:10000]
    # df = df[:50]
    train_num = int(0.7*len(df))
    val_num = int(0.15*len(df))
    train_df = df.iloc[:train_num]
    
    print(train_num, val_num)
    val_df = df.iloc[train_num+1:train_num + val_num]

    test_df = df.iloc[train_num+val_num+1:]
    # ds_train = CustomDataset(train_df)
     
    '''
        {
            "speech_data_wind" : np
            "label" : 0
            "date_quantum": []
        }
    '''
        
    # print("Skipped vals", skipped_vals) #635 vals skipped (train) since cannot form a window for them 815 for val skipped 627 skipped for test
    # print(speaker_window_test_pkl[:2][0]['dates'].size())
    speaker_train_pkl = prepare(train_df, 20)
    speaker_test_pkl = prepare(test_df, 20)
    speaker_val_pkl = prepare(val_df, 20)
    print(len(speaker_train_pkl), 'train')
    print(len(speaker_test_pkl), 'test`')
    print(len(speaker_val_pkl), 'val')

    # with open('speaker_test_2.pkl', 'wb') as f:
    #     pickle.dump(speaker_test_pkl, f)
    
    # with open('speaker_train_2.pkl', 'wb') as f:
    #     pickle.dump(speaker_train_pkl, f)
    # with open('speaker_val_2.pkl', 'wb') as f:
    #     pickle.dump(speaker_val_pkl, f)
    

    
    '''
        dic = {
            speech_d1 : npy
            label: 0,1
            date: datetime
        }
        Shuffle the speakers, do not train for one speaker at a time, batch over speakers, than batch over samples,

        speaker_dict = {
           speaker1:  [
                {
                    speech_data,
                    date,
                    label
                },

            ],
            speaker2: [

            ],
        }

        temporaly save the sliding windows
    '''
