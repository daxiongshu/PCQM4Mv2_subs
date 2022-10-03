import sys
gpu = sys.argv[1]
import os
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
import pytorch_lightning as pl
import torch.nn.functional as F
from tqdm import tqdm
import utils
from constants import PATH,MAX_NODE_FEA
from datasets import get_dl,gety
from gnn import DGCN
from ogb.lsc import PCQM4Mv2Evaluator
import torch
from glob import glob
import numpy as np

def gnn_predict(commit, quick_run=False, logger=True, yaml_path='../yaml/gnn.yaml',
          config=None):

    if config is None:
        config = utils.load_yaml(yaml_path)
    print(config)

    train_dl,valid_dl,testdev_dl,testchallenge_dl,num_feas = get_dl(PATH,config,quick_run)
    print(f'train: {len(train_dl)} valid: {len(valid_dl)}')
    print(f'testdev: {len(testdev_dl)} testchallenge: {len(testchallenge_dl)}')

    model = DGCN(MAX_NODE_FEA, num_feas, config)

    ckpts = glob(f'{PATH}/rnn/{commit}/*/*/*/*.ckpt')
    trainer = pl.Trainer(gpus=1)
    y = gety(valid_dl)
    
    y1,y2,y3 = 0,0,0

    evaluator = PCQM4Mv2Evaluator()
    for ckpt in ckpts:
        print('Predict with {ckpt}')
        yp_valid = trainer.predict(model, valid_dl, ckpt_path=ckpt)
        yp_valid = torch.cat(yp_valid,dim=0)
        y1 += yp_valid

        input_dict = {'y_pred': yp_valid, 'y_true': y}
        result_dict = evaluator.eval(input_dict)
        score2 = result_dict['mae']
        print(f'Valid MAE: {score2:.4f}')

        yp_testdev = trainer.predict(model, testdev_dl, ckpt_path=ckpt)
        yp_testdev = torch.cat(yp_testdev,dim=0)
        y2 += yp_testdev

        yp_testchallenge = trainer.predict(model, testchallenge_dl, ckpt_path=ckpt)
        yp_testchallenge = torch.cat(yp_testchallenge,dim=0)
        y3 += yp_testchallenge

    y1 = y1/len(ckpts)
    y2 = y2/len(ckpts)
    y3 = y3/len(ckpts)

    input_dict = {'y_pred': y1, 'y_true': y}
    result_dict = evaluator.eval(input_dict)
    score2 = result_dict['mae']
    print(f'Ave Valid MAE: {score2:.4f}')

    np.save(f'{PATH}/rnn/{commit}/valid.npy', y1.detach().cpu().float().numpy())
    np.save(f'{PATH}/rnn/{commit}/testdev.npy', y2.detach().cpu().float().numpy())
    np.save(f'{PATH}/rnn/{commit}/testchallenge.npy', y3.detach().cpu().float().numpy())

    msgs = {'VALID MAE':score2}
    return msgs


if __name__ == "__main__":
    quick_run = False
    commit = 'a81b8a4'
    yaml_path=f'../yaml/{commit}.yaml'
    gnn_predict(commit=commit,quick_run=quick_run,yaml_path=yaml_path)
