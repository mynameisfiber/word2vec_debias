# word2vec debias

Playing around with word2vec debiasing as per https://arxiv.org/abs/1607.06520

NOTE: This repo, as it stands, does not work. The solvers included in
[debias.py]() require too much memory. In order to overcome this, we need a
mirror descent solver to solve the optimization problem. Pull requests welcome!

## Requirements

- >=python3.5
- packages in `requirements.txt`
