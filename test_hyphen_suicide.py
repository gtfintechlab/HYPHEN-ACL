import json
import pickle

import torch
import yaml
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from dataset_suicide import SuicidalDataset
from model_hyphen import HYPHEN
from train_hyphen_suicide import eval_loop,get_mcc_score

with open("parameters_suicide.yaml", "r") as f:
    args = yaml.load(f, Loader=yaml.FullLoader)


def main():
    with open("data/test_enc.pkl", "rb") as test:
        df_test = pickle.load(test)

    model_dict = torch.load(args["model_dir"])
    BATCH_SIZE = model_dict["BATCH_SIZE"]
    HIDDEN_DIM = model_dict["HIDDEN_DIM"]
    INPUT_DIM = args["input_size"]
    print(BATCH_SIZE, HIDDEN_DIM, "HYPER")
    ds_test = SuicidalDataset(
        df_test.label.values,
        df_test.enc.values,
        df_test.hist_dates.values,
        window=args["lookback_days"],
        unit=args['unit'],
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    model = HYPHEN(
        INPUT_DIM,
        HIDDEN_DIM,
        BATCH_SIZE,
        attn_type=args["attn_type"],
        learnable_curvature=args["learnable_curvature"],
        init_curvature_val=args["init_curvature_value"],
    ).to(device)
    print(model_dict.keys())
    best_model_wts = model_dict['best_model_wts']
    model.load_state_dict(best_model_wts)
    # total number of parameters in the model
    print("Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    _, _, pred, targets = eval_loop(model, dl_test, device, len(ds_test))
    report = classification_report(targets, pred, labels=[0, 1], output_dict=True)
    mcc = get_mcc_score(targets, pred)
    print(f'MCC is {mcc}')
    print(report)
    model_name = args["model_dir"].split("/")[-1].split(".pt")[0]
    with open(f'{args["results_dir"]} + {model_name} +{mcc}+ .json', "w") as f:
        json.dump(report, f)


if __name__ == "__main__":
    main()
