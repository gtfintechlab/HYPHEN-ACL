import functools
import logging
import pickle
import sys
import warnings

import geoopt.manifolds.stereographic.math as pmath_geo
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AdamW

from earlystopping import EarlyStopping
from model_hyphen import HYPHEN, HypHawkes, TimeLSTMHyp
from nets import MobiusGRU
from radam import RiemannianAdam

torch.manual_seed(0)
# get random value b/w -1 and 1
def get_random_value():
    return 2 * torch.rand(1) - 1

params = {
    "lr": 0.001,
    "epochs": 50,
    "seed": 2020,
    "decay": 1e-5,
    "batch_size": 512,
    "input_size": 768,
    "hidden_size": 512,
    "learnable_curvature": True,
    "init_curvature_val": 0.5,
    "adam_normal": False,
}
device = torch.device("cuda")
warnings.filterwarnings("ignore")
logging.basicConfig(
    filename=f"/root/sanchit/research-group/logs/train_cse-bs-{params['batch_size']}-init-val-{params['init_curvature_val']}-adam-normal-{params['adam_normal']}.log",
    filemode="w",
    level=logging.INFO,
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.log(logging.INFO, f"Running with params: {params}")
train_data_path = "/root/sanchit/research-group/cse_ds/train_data_chinese.pkl"
# val_data_path = 'gdrive/MyoDrive/data/test_data_chinese.pkl'
val_data_path = "/root/sanchit/research-group/cse_ds/test_data_chinese.pkl"


class BaseModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bs,
        device,
        learnable_curvature=False,
        init_curvature_val=0.0,
    ):
        super().__init__()
        self.device = device
        self.hyp_lstm = TimeLSTMHyp(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        # self.linear3 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.3)
        if learnable_curvature:
            self.c = torch.nn.Parameter(torch.tensor([init_curvature_val]).to("cuda"))
        else:
            self.c = torch.FloatTensor([1.0]).to("cuda")
        self.hidden_size = hidden_size
        self.attention = HypHawkes(hidden_size, bs)  # Hawkes and temporal attn

        self.cell = functools.partial(MobiusGRU, k=self.c)
        self.cell_source = self.cell(hidden_size, hidden_size, 1)
        # self.bs = 0

    def init_hidden(self, bs):
        h = (torch.zeros(bs, self.hidden_size, requires_grad=True)).to("cuda")
        c = (torch.zeros(bs, self.hidden_size, requires_grad=True)).to("cuda")

        return (h, c)

    def init_hidden_normal(self, bs):
        h = (torch.zeros(1, bs, self.hidden_size, requires_grad=True)).to("cuda")
        c = (torch.zeros(1, bs, self.hidden_size, requires_grad=True)).to("cuda")

        return (h, c)

    def forward(self, inputs, time_feats):
        """
        inputs: sentence features (B*5*30*N),
        time_feats: (B*5*30)
        """
        bs, lookback, max_tweets, embed_size = inputs.shape
        time_feats = time_feats.permute(1, 0, 2)
        timestamps = []

        for i in range(lookback):
            temp_t = torch.full((bs, max_tweets), (1 / 24) * (i + 1)).to(self.device)
            timestamps.append(time_feats[i] + temp_t)

        timestamps = torch.stack(timestamps).permute(1, 0, 2).to(self.device)
        timestamps = timestamps.reshape(bs, lookback * max_tweets)
        timestamps_lstm = timestamps.detach().clone()
        timestamps.pow_(-1)
        inputs = inputs.reshape(bs, lookback * max_tweets, embed_size)

        # bs = inputs.shape[0]
        h_init, c_init = self.init_hidden(bs)
        h0, c0 = self.init_hidden_normal(lookback * max_tweets)
        # inputs = pmath_geo.expmap0(inputs, k=self.c)
        # inputs = pmath_geo.project(inputs, k = self.c)
        # timestamps = pmath_geo.expmap0(timestamps, k =  self.c)
        # timestamps = pmath_geo.project(timestamps, k = self.c)

        output, (_, _) = self.hyp_lstm(
            inputs, timestamps_lstm, (h_init, c_init), self.c
        )
        # output, _ = self.lstm(inputs, (h0, c0))
        # print(f'lstm out: {output[0:4]}')
        context, output = self.cell_source(output.permute(1, 0, 2))
        output = output.permute(1, 0, 2)
        context = context.permute(1, 0, 2)
        # print(f'cell out: {output[0:4]}, {context[0:4]}')
        # print(context.shape,'context')
        # print(output.shape,'output')
        # output = output[-1]
        output = pmath_geo.logmap0(pmath_geo.project(output, k=self.c), k=self.c)
        context = pmath_geo.logmap0(pmath_geo.project(context, k=self.c), k=self.c)
        # output_fin = output
        # print(output.shape,'outpu2')
        output_fin, _ = self.attention(output, context, timestamps, c=self.c)
        # print(f'attention out: {output_fin[0:4]}')
        # print(output_fin.shape,'output')
        output_fin = output_fin.permute(1, 0, 2)
        output_fin = output_fin.squeeze(0)
        output_fin = self.linear1(output_fin)
        # output_fin = F.relu(output_fin)
        output_fin = self.dropout(output_fin)
        cse_output = self.linear2(output_fin)
        # margin_output = self.linear3(output_fin)
        return cse_output
        # output = self.dropout(F.relu(self.linear))


class FinCLData(Dataset):
    """"""

    def __init__(self, data_path):
        """
        data_path: path to the data pickle file.
        """
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        temp = self.data[idx]
        # embeds = temp["embedding"]
        # movement_label = temp["movement_label"]
        # volatility = temp["volatility"]

        return temp


def main():
    traindata = FinCLData(train_data_path)
    valdata = FinCLData(val_data_path)
    trainloader = torch.utils.data.DataLoader(
        traindata,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )
    valloader = torch.utils.data.DataLoader(
        valdata,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=8,
        drop_last=True,
    )
    dataloaders = {"train": trainloader, "val": valloader}
    criterion = nn.MSELoss()
    loss_history = {"train": [], "val": []}
    accuracy_history = {"train": [], "val": []}
    mcc_history = {"train": [], "val": []}
    f1_history = {"train": [], "val": []}
    model = BaseModel(
        input_size=params["input_size"],
        hidden_size=params["hidden_size"],
        bs=params["batch_size"],
        device=device,
        learnable_curvature=params["learnable_curvature"],
        init_curvature_val=params["init_curvature_val"],
    ).to(device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    if not params["adam_normal"]:
        optimizer = RiemannianAdam(
            model.parameters(), lr=params["lr"], weight_decay=params["decay"]
        )
    else:
        optimizer = AdamW(
            model.parameters(), lr=params["lr"], weight_decay=params["decay"]
        )
    # optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2
    )

    early_stopping = EarlyStopping(patience=7, verbose=True)

    for epoch in tqdm(range(params["epochs"])):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            truelabels = []
            predlabels = []
            # Iterate over data.
            for batch_data in dataloaders[phase]:
                embedding_data = batch_data["embedding"]
                # print ('Embedding data: ', embedding_data.shape)
                # embedding_data = embedding_data.type(torch.DoubleTensor).to(device)
                embedding_data = embedding_data.to(device)

                target = batch_data["volatility"]
                target[torch.isnan(target)] = 0
                target[torch.isinf(target)] = 0
                target = target.type(torch.FloatTensor).to(device).unsqueeze(-1)

                length = batch_data["length_data"]

                time_feats = batch_data["time_feature"].to(device).squeeze(-1)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(embedding_data, time_feats)

                    # print ('Outputs: ', outputs[0:4])
                    # print ('Targets: ', target[0:4])
                    cse_loss = criterion(outputs, target)
                    loss = cse_loss
                    running_loss += loss.item()

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

            epoch_loss = running_loss / len(dataloaders[phase])
            loss_history[phase].append(epoch_loss)

            if phase == "val":
                early_stopping(epoch_loss, model)
                scheduler.step(epoch_loss)
                # torch.save(
                #     {
                #         "model_wts": model.state_dict(),
                #         "current_epoch": epoch,
                #         "loss_history": loss_history,
                #     },
                #     save_path + "vol_model_stock_china_500_3.pth",
                # )

                logger.info(
                    "{} Epoch: {} Loss: {:.4f} ".format(
                        phase,
                        epoch,
                        epoch_loss,
                    )
                )
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break
    save_path = "/root/sanchit/research-group/saved_models_cse"


if __name__ == "__main__":
    main()
