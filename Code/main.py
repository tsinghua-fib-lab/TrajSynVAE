# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
Created on Sep 15th 18:05:53 2020
Main Function
"""
import csv
import datetime
import os
import argparse
import torch
import numpy as np

from data_prepare import MYDATA, getsets, reform
from model import VAE
from lstm import LSTMMODEL
from semi_markov import SEMIMARKOV
from result_evaluation import EVALUATION

torch.set_default_tensor_type(torch.DoubleTensor)


def EVAL(param, data):
    print('Start Evaluation')
    generated = data.GENDATA[-1]
    original = data.REFORM['test']
    np.save(param.save_path + 'data/original.npy', original)
    np.save(param.save_path + 'data/generated.npy', generated)
    jsd = EVALUATION(param, generated, SM, original, 'Our Model', 'Semi Markov', 'plot')
    # Overlapping_ratio_plot(generated, original)
    with open(param.save_path + 'result_evaluation.csv', 'a', encoding='utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['Our', jsd['travel_distance'][0], jsd['radius'][0], jsd['duration'][0],
                            jsd['G_rank'][0], jsd['I_rank'][0], jsd['move'][0], jsd['stay'][0]])
        csv_writer.writerow(['Semi_Markov', jsd['travel_distance'][1], jsd['radius'][1], jsd['duration'][1],
                            jsd['G_rank'][1], jsd['I_rank'][1], jsd['move'][1], jsd['stay'][1]])
    return jsd


class parameters(object):

    def __init__(self, args) -> None:
        super().__init__()
        
        self.model_type = args.model_type
        self.data_type = args.data_type
        self.location_mode = args.location_mode

        self.tim_emb_size = args.tim_emb_size
        self.loc_emb_size = args.loc_emb_size
        self.usr_emb_size = args.usr_emb_size
        self.d_model = args.d_model
        self.encoder_rnn_hidden_size = args.encoder_rnn_hidden_size
        self.z_hidden_size_mean = args.z_hidden_size_mean
        self.z_hidden_size_std = args.z_hidden_size_std
        self.latent_size = args.latent_size
        self.dropout = args.dropout
        self.decoder_rnn_hidden_size = args.decoder_rnn_hidden_size
        self.loc_hidden_size1 = args.loc_hidden_size1
        self.loc_hidden_size2 = args.loc_hidden_size2
        self.tim_hidden_size1 = args.tim_hidden_size1
        self.tim_hidden_size2 = args.tim_hidden_size2
        self.poi_weight = args.poi_weight

        self.learning_rate = args.learning_rate
        self.L2 = args.L2
        self.step_size = args.step_size
        self.gamma = args.gamma
        self.epoches = args.epoches
        self.batchsize = args.batchsize
        self.trainsize = args.trainsize
        self.exptimes = args.exptimes
        self.fourier = args.fourier
        self.poi_ban = args.poi_ban
        self.poi_emb_ban = args.poi_emb_ban
        self.pos_emb_ban = args.pos_emb_ban
        self.feedback_ban =args.feedback_ban

        save_path = './RES/' + '-'.join([str(self.__dict__[v]) for _, v in enumerate(self.__dict__)]) + '/' + str(datetime.datetime.now().strftime('%Y-%m%d-%H%M') + '/0/')
        os.makedirs(save_path)
        param_name = [x for x in self.__dict__]
        param_value = [self.__dict__[v] for v in self.__dict__]
        self.save_path = save_path
        self.param_name = param_name
        self.param_value = param_value

        self.device = torch.device(('cuda:' + args.cuda) if torch.cuda.is_available() else 'cpu')

    def data_info(self, data):
        self.POI = data.POI
        self.GPS = data.GPS
        self.USERLIST = data.USERLIST

        self.loc_size = data.loc_size
        self.tim_size = data.tim_size
        self.usr_size = data.usr_size
        self.poi_size = data.poi_size

        self.infer_maxlast = data.infer_maxlast 
        self.infer_maxinternal = data.infer_maxinternal 
        self.infer_divide = data.infer_divide 
        

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_type', type=str, default='FourSquare_NYC', choices=['ISP', 'GeoLife', 'FourSquare_NYC', 'FourSquare_TKY'])
    parser.add_argument('-l', '--location_mode', type=int, default=1, choices=[0, 1, 2])
    parser.add_argument('-m', '--model_type', type=str, default='VAE', choices=['VAE', 'LSTM'])

    parser.add_argument('--tim_emb_size', type=int, default=256)
    parser.add_argument('--loc_emb_size', type=int, default=256)
    parser.add_argument('--usr_emb_size', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--encoder_rnn_hidden_size', type=int, default=512)
    parser.add_argument('--z_hidden_size_mean', type=int, default=256)
    parser.add_argument('--z_hidden_size_std', type=int, default=256)
    parser.add_argument('--latent_size', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--decoder_rnn_hidden_size', type=int, default=512)
    parser.add_argument('--loc_hidden_size1', type=int, default=128)
    parser.add_argument('--loc_hidden_size2', type=int, default=128)
    parser.add_argument('--tim_hidden_size1', type=int, default=64)
    parser.add_argument('--tim_hidden_size2', type=int, default=64)
    parser.add_argument('--poi_weight', type=float, default=0.1)

    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--L2', type=float, default=1e-5)
    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('-e', '--epoches', type=int, default=120)
    parser.add_argument('-b', '--batchsize', type=int, default=3)
    parser.add_argument('-t', '--trainsize', type=float, default=0.8)
    parser.add_argument('-n', '--exptimes', type=int, default=1)

    parser.add_argument('--cuda', type=str, default='0', choices=['0', '1', '2', '3'])
    parser.add_argument('--fourier', type=bool, default=False)
    parser.add_argument('--poi_ban', type=bool, default=False)
    parser.add_argument('--poi_emb_ban', type=bool, default=False)
    parser.add_argument('--pos_emb_ban', type=bool, default=False)
    parser.add_argument('--feedback_ban', type=bool, default=False)

    args = parser.parse_args()
    param = parameters(args)

    data = MYDATA(param.data_type, param.location_mode)
    param.data_info(data)


    for i in range(param.exptimes):
        trainset, validset, testset = getsets(data, param.trainsize, 0.9 - param.trainsize)
        reform(trainset, 'train')
        reform(validset, 'valid')
        reform(testset, 'test')
    
        SM = SEMIMARKOV(data.REFORM['train'])
        print('Data Loaded')
        param.save_path = param.save_path[:-2] + str(i) + '/'
        os.makedirs(param.save_path + 'plots')
        os.makedirs(param.save_path + 'data')
        with open(param.save_path + 'result_evaluation.csv', 'a', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(param.param_name)
            csv_writer.writerow(param.param_value)
            csv_writer.writerow(['Method', 'travel_distance', 'radius', 'duration', 'G_rank', 'I_Rank', 'move', 'stay'])

        if param.model_type == 'VAE':
            model = VAE(param) 
        else:
            model = LSTMMODEL(param)
        model = model.double().to(param.device)
        model.run(trainset, validset, testset)
        # model.load('./Model')
        # model.test_data_prepare(testset.dataset)
        jsd = EVAL(param, data)
