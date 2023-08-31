"""
Created on Sep 15th 18:05:53 2020

Author: Qizhong Zhang

Main Function
"""

import csv
import datetime
import os
import argparse
import numpy as np
import torch
from torch.utils.data import Subset

from data_prepare import MYDATA, reform
from model import VAE
from lstm import LSTMMODEL
from baselines import SEMIMARKOV, HAWKES, TimeGeo, MoveSim
from result_evaluation import EVALUATION

torch.set_default_tensor_type(torch.DoubleTensor)

# Hyperparameters
class parameters(object):

    def __init__(self, args) -> None:
        super().__init__()
        
        # Data-related
        self.data_type = args.data_type
        self.location_mode = args.location_mode
        self.trainsize = args.trainsize

        # Model-related
        self.model_type = args.model_type
        self.rnn_type = args.rnn_type
        self.rnn_layers = args.rnn_layers
        self.rnn_bidirectional = args.rnn_bidirectional
        self.dual_rnn = args.dual_rnn
        # Embedding
        self.tim_emb_type = args.tim_emb_type
        self.tim_emb_size = args.tim_emb_size
        self.loc_emb_size = args.loc_emb_size
        self.usr_emb_size = args.usr_emb_size
        self.d_model = args.d_model
        # Encoder
        self.encoder_rnn_hidden_size = args.encoder_rnn_hidden_size
        self.z_hidden_size_mean = args.z_hidden_size_mean
        self.z_hidden_size_std = args.z_hidden_size_std
        self.latent_size = args.latent_size
        # Regularization
        self.layernorm = args.layernorm
        self.dropout = args.dropout
        # Decoder-Location
        self.decoder_rnn_hidden_size = args.decoder_rnn_hidden_size
        self.loc_hidden_size1 = args.loc_hidden_size1
        self.loc_hidden_size2 = args.loc_hidden_size2
        self.cdfpoi = args.cdfpoi
        self.poi_weight = args.poi_weight
        self.poi_weight_dynamic = args.poi_weight_dynamic
        self.loc_initial = args.loc_initial
        # Decoder-Time
        self.tim_hidden_size1 = args.tim_hidden_size1
        self.tim_hidden_size2 = args.tim_hidden_size2        
        self.time_initial = args.time_initial

        # KL-annealing
        self.max_beta = args.max_beta
        self.cycle = int(args.cycle * 32 / args.batchsize)
        # Learning
        self.learning_rate = args.learning_rate
        self.L2 = args.L2
        self.step_size = args.step_size
        self.gamma = args.gamma
        self.epoches = args.epoches
        self.batchsize = args.batchsize
        # Generating
        self.ntrajs = args.ntrajs
        self.first_sample = args.first_sample
        # Unbalanced learning
        self.tim_only = args.tim_only
        self.loc_only = args.loc_only
        # Ablation
        self.fourier = args.fourier
        self.poi_ban = args.poi_ban
        self.poi_emb_ban = args.poi_emb_ban
        self.pos_emb_ban = args.pos_emb_ban
        self.feedback_ban =args.feedback_ban

        # checkpoint
        self.save_path = './RES/' + str(datetime.datetime.now().strftime('%Y-%m%d-%H%M') + '/0/')
        self.param_name = [x for x in self.__dict__]
        self.param_value = [self.__dict__[v] for v in self.__dict__]

        # Modes
        self.tune = args.tune
        self.exptimes = args.exptimes
        self.checkpoint = args.checkpoint
        self.generate = args.generate
        self.run_baselines = args.run_baselines

        # CUDA
        self.cuda = args.cuda
        self.device = torch.device(('cuda:' + args.cuda) if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')

    # Data-related information
    def data_info(self, data):
        self.POI = data.POI
        self.GPS = data.GPS
        self.USERLIST = data.USERLIST

        self.loc_size = data.loc_size
        self.tim_size = data.tim_size
        self.usr_size = data.usr_size
        self.poi_size = data.poi_size

        self.infer_maxlast = data.infer_maxlast 
        

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()

    # Data-related
    parser.add_argument('-d', '--data_type', type=str, default='ISP', choices=['ISP', 'GeoLife', 'FourSquare_NYC', 'FourSquare_TKY'])
    parser.add_argument('-l', '--location_mode', type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('-t', '--trainsize', type=float, default=0.9)

    # Model-related
    parser.add_argument('-m', '--model_type', type=str, default='VAE', choices=['VAE', 'LSTM'])
    parser.add_argument('--rnn_type', type=str, default='LSTM', choices=['LSTM', 'GRU'])
    parser.add_argument('--rnn_layers', type=int, default=1)
    parser.add_argument('--rnn_bidirectional', type=bool, default=False)
    parser.add_argument('--dual_rnn', type=bool, default=False)
    # Embedding
    parser.add_argument('--tim_emb_type', type=str, default='Linear', choices=['Linear', 'Categorical'])
    parser.add_argument('--tim_emb_size', type=int, default=256)
    parser.add_argument('--loc_emb_size', type=int, default=256)
    parser.add_argument('--usr_emb_size', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=256)
    # Encoder
    parser.add_argument('--encoder_rnn_hidden_size', type=int, default=512)
    parser.add_argument('--z_hidden_size_mean', type=int, default=256)
    parser.add_argument('--z_hidden_size_std', type=int, default=256)
    parser.add_argument('--latent_size', type=int, default=512)
    # Regularization
    parser.add_argument('--layernorm', action='store_false')
    parser.add_argument('--dropout', type=float, default=0.2)
    # Decoder-Location
    parser.add_argument('--decoder_rnn_hidden_size', type=int, default=512)
    parser.add_argument('--loc_hidden_size1', type=int, default=128)
    parser.add_argument('--loc_hidden_size2', type=int, default=128)
    parser.add_argument('--cdfpoi', action='store_false')
    parser.add_argument('--poi_weight', type=float, default=0.1)
    parser.add_argument('--poi_weight_dynamic', type=bool, default=False)
    parser.add_argument('--loc_initial', type=float, default=1e9)
    # Decoder-Time
    parser.add_argument('--tim_hidden_size1', type=int, default=128)
    parser.add_argument('--tim_hidden_size2', type=int, default=128)
    parser.add_argument('--time_initial', type=float, default=100)

    # KL-annealing
    parser.add_argument('--max_beta', type=float, default=0.5) # 1
    parser.add_argument('--cycle', type=int, default=500) # 100
    # Learning
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--L2', type=float, default=1e-5)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('-e', '--epoches', type=int, default=50)
    parser.add_argument('-b', '--batchsize', type=int, default=32)
    # Unbalanced learning
    parser.add_argument('--tim_only', type=bool, default=False)
    parser.add_argument('--loc_only', type=bool, default=False)
    # Generating
    parser.add_argument('--ntrajs', type=int, default=7)
    parser.add_argument('--first_sample', type=str, default='New', choices=['New', 'Original'])
    # Ablation
    parser.add_argument('--fourier', type=bool, default=False)
    parser.add_argument('--poi_ban', type=bool, default=False)
    parser.add_argument('--poi_emb_ban', type=bool, default=False)
    parser.add_argument('--pos_emb_ban', type=bool, default=False)
    parser.add_argument('--feedback_ban', type=bool, default=False)

    # Modes
    parser.add_argument('-n', '--exptimes', type=int, default=1)
    parser.add_argument('--tune', type=bool, default=False)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--generate', type=bool, default=False)
    parser.add_argument('--run_baselines', type=bool, default=False)
    parser.add_argument('--cuda', type=str, default='0', choices=['0', '1', '2', '3'])

    # hyperparameters and dataset
    args = parser.parse_args()
    param = parameters(args)

    data = MYDATA(param.data_type, param.location_mode)
    param.data_info(data)

    trainid, validid, testid = data.split(validprop=0.9 - param.trainsize)
    trainset, validset, testset = Subset(data, trainid), Subset(data, validid), Subset(data, testid)

    reform(trainset, 'train')
    reform(testset, 'test')

    # Logging
    os.makedirs(param.save_path[:-2])
    with open(param.save_path[:-2] + 'result.csv', 'a', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(param.param_name)
        csv_writer.writerow(param.param_value)
        csv_writer.writerow(['Method', 'travel_distance', 'radius', 'duration', 'G_rank', 'move', 'stay'])

    # Baselines
    SM = []
    if param.run_baselines:
        HK, TG, MS, jsd = [], [], [], []

    for i in range(param.exptimes):

        # Baselines
        SM.append(SEMIMARKOV(data.REFORM['train'], param.tim_size))
        if param.run_baselines:
            HK.append(HAWKES(data.REFORM['train'], param))
            TG.append(TimeGeo(data.REFORM['train'], param))
            MS.append(MoveSim(data, param, i))
            jsd1 = EVALUATION(param, SM[i], HK[i], data.REFORM['test'], 'Semi Markov', 'Hawkes', 'jsd')
            jsd2 = EVALUATION(param, TG[i], MS[i], data.REFORM['test'], 'TimeGeo', 'MoveSim', 'jsd')
            jsd.append(np.concatenate([jsd1.values, jsd2.values], axis=0))
            continue
        print('Data Loaded')

        # save_path
        param.save_path = param.save_path[:-2] + str(i) + '/'
        os.makedirs(param.save_path + 'plots')
        os.makedirs(param.save_path + 'data')

        # Model initialization
        if param.model_type == 'VAE':
            model = VAE(param) 
        else: 
            model = LSTMMODEL(param)
        model = model.double().to(param.device)
        if param.checkpoint is not None:
            model.load(param.checkpoint)

        # Run model
        if param.generate:
            if param.model_type =='VAE':
                model.initial_prob = model.inference_initial(trainset)
            model.user_indicator, model.loc_weights = model.location_constraints(trainset.dataset.REFORM['train'])
            model.test_data_prepare(testset.dataset)
        else:
            model.run(trainset, testset)

        # Save results
        np.save(param.save_path + 'data/original.npy', data.REFORM['test'])
        np.save(param.save_path + 'data/Semi_Markov.npy', SM)
        
    # Evaluation and Logging
    print('Start Evaluation')
    jsdtotal = []
    for i in range(len(data.GENDATA)):
        jsdtotal.append(EVALUATION(param, data.GENDATA[i], SM[i], data.REFORM['test'], 'Our Model', 'Semi Markov', 'plot'))
        with open(param.save_path[:-2] + 'result.csv', 'a', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['Semi_Markov', jsdtotal[-1]['travel_distance'][1], jsdtotal[-1]['radius'][1], jsdtotal[-1]['duration'][1],
                                jsdtotal[-1]['G_rank'][1], jsdtotal[-1]['move'][1], jsdtotal[-1]['stay'][1]])
            csv_writer.writerow(['Our', jsdtotal[-1]['travel_distance'][0], jsdtotal[-1]['radius'][0], jsdtotal[-1]['duration'][0],
                                jsdtotal[-1]['G_rank'][0], jsdtotal[-1]['move'][0], jsdtotal[-1]['stay'][0]])

    # Important Statistics
    result = np.array([jsdtotal[i].values for i in range(len(jsdtotal))]) if not param.run_baselines else np.array(jsd)
    for i in range(result.shape[1]):
        print(result[:, i, :].mean(0), result[:, i, :].std(0))
