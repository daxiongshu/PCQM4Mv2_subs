## PCQM4Mv2_subs

This repo contains our submission to the [NeurIPS 2022 OGB LSC PCQM4Mv2 challenge](https://ogb.stanford.edu/docs/lsc/leaderboards/#pcqm4mv2).

### Install
- pip install -U torch ogb rdkit
- pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu102.html
- pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cu102.html
- pip install torch-geometric

### Train
- cd scripts
- sh train.sh

### Predict
- cd scripts
- python predict.py 0
