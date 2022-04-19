# Hyperbolic Hawkes Attention Network For Sequence Modeling

This codebase contains the python scripts for HYPHEN, the model for the ACL 2022 paper, "Hyperbolic Hawkes Attention Network For Sequence Modeling
". This work was done with the FinTech lab at Georgia Tech. The FinTech lab aims to be a hub for finance education, research and industry in the Southeast. The lab acts as a platform to connect and bring together faculty and students across Georgia Tech with the financial services industry and FinTech entrepreneurs. 


## HYPHEN

This codebase contains the python scripts for HYPHEN.

## Environment and setup
Dependencies are mentioned in the requirements.txt. Just run the below shell script to setup your environment. Please ensure that the data is in the current directory as of the `train_hyphen.py` file.
### Set env
``` ./set_env.sh ```


## YAML files
We have yaml files for the datasets that we experimented with and can be referred to for the different scripts. Please modify the yaml files accordingly for train/test scripts.

### Train your model
```
python train_hyphen_suicide.py
```
### Testing your model
```
python test_hyphen_suicide.py
```

## Cite
Consider citing our work if you use our codebase

```c
@inproceedings{Agarwal-etal-2022,
    title = "Hyperbolic Hawkes Attention Network For Sequence Modeling",
    author = "Agarwal, Shivam  and 
      Sawhney, Ramit  and
      Ahuja, Sanchit, and
      Soun, Ritesh and 
      Chava, Sudheer"
    booktitle = "Proceedings of the 60th Annual Meeting of The Association of Computational Linguistics",
    month = may,
    year = "2022",
    address = "Dublin",
    publisher = "Association for Computational Linguistics"}
```

## Datasets

### Europarl Vote
We use the dataset released by [1] that consist of parliamentry debates, date of the debate, vote of the MP etc. We use a static dump of BERT embeddings for parliamentery debates
stored in a `npy` format along with the europarl dataset.  

### Suicide Ideation
For Suicide Ideation, we followed the exact way of preprocessing and preparation of data as done [here](https://github.com/midas-research/sismo-wsdm)

### Stock Datasets
We have used couple of datasets for stock price prediction, i.e. Chinese Stock Exchange and S&P dataset. We follow exactly the same preprocessing as done here [7]. 


## References
1. Abercrombie, Gavin (2020), “ParlVote:  Corpora for Sentiment Analysis of Political Debatess”, Mendeley Data, V2, doi: 10.17632/czjfwgs9tm.2
2. Ramit Sawhney, Harshit Joshi, Saumya Gandhi, and Rajiv Ratn Shah. 2021. Towards Ordinal Suicide Ideation Detection on Social Media. Proceedings of the 14th ACM International Conference on Web Search and Data Mining. Association for Computing Machinery, New York, NY, USA, 22–30. DOI:https://doi.org/10.1145/3437963.3441805
3. Kochurov, Max, Rasul Karimov, and Serge Kozlukov. "Geoopt: Riemannian optimization in pytorch." arXiv preprint arXiv:2005.02819 (2020).
4. Hyrnn code: https://github.com/ferrine/hyrnn
5. Manifolds and RAdam optimizer: https://github.com/HazyResearch/hgcn
6. Ramit Sawhney, Shivam Agarwal, Megh Thakkar, Arnav Wadhwa, and Rajiv Ratn Shah. 2021. Hyperbolic Online Time Stream Modeling. Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. Association for Computing Machinery, New York, NY, USA, 1682–1686. DOI:https://doi.org/10.1145/3404835.3463119
7. https://github.com/midas-research/fast-eacl
