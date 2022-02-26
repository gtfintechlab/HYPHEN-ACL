import torch
import math
import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix
from preprocess import prepare
from model import HYPHEN

parser = argparse.ArgumentParser(description="HYPHEN")

parser.add_argument(
    "--model_path",
    type=str,
    default = 'saved_models/-epoch50-vanillalookback-5.pth',
    help="Path to saved model checkpoint",
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
    default=5,
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


def test(model, speakers_test):

    model.eval()
    with torch.no_grad():
        total_correct_test = []
        test_labels = []
        loss_test = 0.0
        total_test = 0
        for ds_test in speakers_test:
            input_tensor_test = ds_test['speech_data']
            label_test = ds_test['label']
            label_test = label_test.long().squeeze(1)
            test_labels.extend(label_test.tolist())
            dates = ds_test['dates'].squeeze(1)
            dates_inv = ds_test['dates_inv'].squeeze(1)

            output = model(input_tensor_test.squeeze(1),
                           dates, dates_inv).squeeze(1)
            pred = torch.argmax((output), -1).tolist()
            total_correct_test.extend(pred)
            total_test += len(output)
        confus_matrix_test = confusion_matrix(
            test_labels, total_correct_test)
        tn, fp, fn, tp = confus_matrix_test.ravel()
        tacc = (tp+tn)/(tp+tn+fp+fn)
        tf1 = (tp)/(tp+0.5*(fp+fn))
        tmcc = (tp*tn-fp*fn)/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
        print(
            f"Tmcc for this model {args.model_path} is: {tmcc} and tf1 is {tf1} and accuracy is {tacc}")


if __name__ == "__main__":
    df = pd.read_csv('ParlVote_concat.csv')
    train_num = int(0.7*len(df))
    val_num = int(0.15*len(df))
    train_df = df.iloc[:train_num]
    val_df = df.iloc[train_num+1:train_num + val_num]
    test_df = df.iloc[train_num+val_num+1:]
    speakers_test = prepare(test_df, window=args.lookback_days)
    model = HYPHEN(768, args.batch_size, args.batch_size,
                   attn_type=args.attn_type, learnable_curvature=args.learnable_curvature, init_curvature_val=args.init_curvature_val).to('cuda')
    model.load_state_dict(
        torch.load(args.model_path)["model_wts"]
    )
    test(model, speakers_test)
