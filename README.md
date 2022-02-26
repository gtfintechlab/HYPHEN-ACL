# Hyperbolic Hawkes Attention Network For Sequence Modeling

This codebase contains the python scripts for HYPHEN, the model for the ACL 2022 paper, "Hyperbolic Hawkes Attention Network For Sequence Modeling
". This work was done with the FinTech lab at Georgia Tech. The FinTech lab aims to be a hub for finance education, research and industry in the Southeast. The lab acts as a platform to connect and bring together faculty and students across Georgia Tech with the financial services industry and FinTech entrepreneurs. 


## HYPHEN

Dependencies are mentioned in the requirements.txt. Just run the below shell script to setup your environment. Datasets can be obtained [here](https://drive.google.com/file/d/1PXAW5oNLDu1ceiiRfIk6EiT0ei-noKff/view?usp=sharing). Please ensure that the data is in the current directory as of the `train_fin.py` file. There are fine-tuned BERT embeddings in speech_numpys folder and a Parlvote_concat.csv file.
### Set env
``` ./set_env.sh ```

## Train HYPHEN
```
python train_fin.py --lookback_days 5 --attn_type hyp_hawkes --learnable_curvature True

```
## Test HYPHEN
```
python test.py --model_path "Your model path" --lookback_days 5 --attn_type hyp_hawkes --learnable_curvature True
```

## Cite
Consider citing our work if you use our codebase

```c
@inproceedings{Agarwal-etal-2022,
    title = "Hyperbolic Hawkes Attention Network For Sequence Modeling",
    author = "Agarwal, Shivam  and 
      Sawhney, Ramit  and
      Ahuja, Sanjit, and
      Soun, Ritesh and 
      Chava, Sudheer"
    booktitle = "Proceedings of the 60th Annual Meeting of The Association of Computational Linguistics",
    month = may,
    year = "2022",
    address = "Dublin",
    publisher = "Association for Computational Linguistics"}
```

