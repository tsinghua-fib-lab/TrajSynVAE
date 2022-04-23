# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
Created on Sep 15th 18:05:53 2020

Baseline 3: Hawkes
"""
import neurawkes_master.train_models
import neurawkes_master.generate_sequences
import numpy as np
import pandas as pd
import pickle
import argparse


def ToHawkes(data):
    trajs = pd.concat([pd.DataFrame.from_dict(data[traj]) for traj in data]).reset_index(drop=True)
    trajs['idx_event'] = trajs.index
    VALIDID = np.sort(trajs['loc'].unique())
    trajs['loc'] = trajs['loc'].apply(lambda x:np.where(VALIDID == x)[0][0])
    trajs['sta'] = trajs['tim'] - trajs['tim'].shift(1)
    trajs['tim'] -= trajs['tim'].min()
    def lastsame(d):
        d['time_since_last_same_event'] = (d['tim'] - d['tim'].shift(1)).fillna(d['tim'].iloc[0])
        return d
    trajs = trajs.groupby('loc').apply(lastsame).reset_index(drop=True)
    trajs = trajs.rename(columns={'tim': 'time_since_start', 'loc': 'type_event', 'sta': 'time_since_last_event'})
    return trajs.iloc[1:], VALIDID


def HAWKESPROCESS(data, PATH, MODE):

    Hawkes = {}
    VALIDIDS = {}

    for user in data.USERLIST:

        trajs, VALIDID = ToHawkes(data.DATA[user])

        VALIDIDS[user] = VALIDID
        Hawkes[user] = {'train': list(trajs.iloc[:int(trajs.shape[0] * 0.8)].to_dict(orient='index').values()), 
                            'dev': list(trajs.iloc[int(trajs.shape[0] * 0.8):int(trajs.shape[0] * 0.9)].to_dict(orient='index').values()),
                            'test': list(trajs.iloc[int(trajs.shape[0] * 0.9):].to_dict(orient='index').values()), 
                            'dim_process': VALIDID.shape[0], 
                            'args': {}}

    with open('./data/' + PATH + '/hawkes_' + MODE + '.pkl', 'wb') as f:
        pickle.dump(Hawkes, f)

    with open('./data/' + PATH + '/hawkes_validid_' + MODE + '.pkl', 'wb') as f:
        pickle.dump(VALIDIDS, f)


def hawkes(PATH, MODE):
    
    # Generate trajectories for each user
    with open('./data/' + PATH + '/hawkes_' + MODE + '.pkl', 'rb') as f:
        Hawkes = pickle.load(f)
    
    with open('./data/' + PATH + '/hawkes_validid_' + MODE + '.pkl', 'rb') as f:
        VALIDIDS = pickle.load(f)

    output = {}
    for user in Hawkes:

        print('-' * 160)
        print('INFERENCING FOR USER:' + str(user))

        with open('./data/train.pkl', 'wb') as f:
            pickle.dump(Hawkes[user], f)

        with open('./data/dev.pkl', 'wb') as f:
            pickle.dump(Hawkes[user], f)

        with open('./data/test.pkl', 'wb') as f:
            pickle.dump(Hawkes[user], f)

        # Train
        parser = argparse.ArgumentParser(description='Trainning model ... ')
        parser.add_argument('-m', '--Model', type=str, default='hawkes',
                            choices=['hawkes', 'hawkesinhib', 'conttime'],
                            help='Which model to train? hawkes (SE-MPP)? hawkesinhib (D-SM-MPP)? conttime (N-SM-MPP)?')
        parser.add_argument('-fd', '--FileData', type=str, default='./data/',
                            help='Path of the dataset (e.g. ./data/data_hawkes/)')
        parser.add_argument('-tr', '--TrainRatio', default=1.0, type=float, help='How much data to train?')
        parser.add_argument('-cl2', '--CoefL2', default=0.0, type=float, help='Coefficient of L2 norm')
        parser.add_argument('-d', '--DimLSTM', default=64, type=int, help='Dimension of LSTM model ')
        parser.add_argument('-s', '--Seed', default=12345, type=int, help='Seed of random state')
        parser.add_argument('-fp', '--FilePretrain', required=False,
                            help='File of pretrained model (e.g. ./tracks/track_PID=XX_TIME=YY/model.pkl)')
        parser.add_argument('-tp', '--TrackPeriod', default=1, type=int, help='Track period of training')
        parser.add_argument('-me', '--MaxEpoch', default=1, type=int, help='Max epoch number of training')
        parser.add_argument('-sb', '--SizeBatch', default=1, type=int, help='Size of mini-batch')
        parser.add_argument('-op', '--Optimizer', default='adam', type=str, choices=['adam', 'sgd'],
                            help='Optimizer of training')
        parser.add_argument('-mt', '--MultipleTrain', default=1, type=int,
                            help='Multiple of events to sample (integral) for training')
        parser.add_argument('-md', '--MultipleDev', default=10, type=int,
                            help='Multiple of events to sample (integral) for dev')
        parser.add_argument('-wt', '--WhatTrack', default='loss', type=str, choices=['loss', 'rmse', 'rate'],
                            help='What to track for early stoping ? ')
        parser.add_argument('-ls', '--LossType', default='loglikehood', type=str,
                            choices=['loglikehood', 'prediction'],
                            help='What is the loss to optimized ?')
        parser.add_argument('-lr', '--LearnRate', default=0.00001, type=float, help='What learning rate to use ?')
        parser.add_argument('-pp', '--PartialPredict', default=0, type=int, choices=[0, 1],
                            help='What to only predict part of stream ? 0--False, 1--True')
        parser.add_argument('-ps', '--PruneStream', default=0, type=int,
                            help='Prune stream? Give me the index ! 0 is nothing to prune. Note : index specifies a COMBINATION of event types by its binary coding (e.g. 0--00000, 1--00001, 31-11111 where 1 means this type is pruned)!')
        parser.add_argument('-ds', '--DevIncludedSetting', default=0, type=int, choices=[0, 1],
                            help='Alternative setting (fix tuned hyper-params, train on combo of train and dev, then test)? 0--False, 1--True Note: in our project, this is ONLY used to compare prev work on MIMIC, SO and Financial datasets')
        parser.add_argument('-pf', '--PredictFirst', default=1, type=int, choices=[0, 1],
                            help='Predict the first event ? 0--False, 1--True Note: in our project, this is False ONLY on MIMIC, SO and Financial datasets')
        parser.add_argument('-pl', '--PredictLambda', default=0, type=int, choices=[0, 1],
                            help='Predict Lambda (intensity) ? 0--False, 1--True Note: this is used ONLY in intensity evaluation')
        args = parser.parse_args()
        path_save = neurawkes_master.train_models.main(args)

        # Generate
        parser = argparse.ArgumentParser(description='Generating sequences... ')
        parser.add_argument('-m', '--ModelGen', default='hawkes', type=str,
                            choices=['hawkes', 'hawkesinhib', 'conttime'],
                            help='Model used to generate data')
        parser.add_argument('-sp', '--SetParams', default=0, type=int, choices=[0, 1],
                            help='Do we set the params ? 0 -- False; 1 -- True')
        parser.add_argument('-st', '--SumForTime', default=1, type=int, choices=[0, 1],
                            help='Do we use total intensity for time sampling? 0 -- False; 1 -- True')
        parser.add_argument('-fp', '--FilePretrain', default=path_save + 'model.pkl', type=str,
                            help='File of pretrained model (e.g. ./tracks/track_PID=XX_TIME=YY/model.pkl)')
        parser.add_argument('-s', '--Seed', default=12345, type=int, help='Seed of random state')
        parser.add_argument('-k', '--DimProcess', default=len(VALIDIDS[user]), type=int, help='Number of event types')
        parser.add_argument('-d', '--DimLSTM', default=32, type=int, help='Dimension of LSTM generator')
        parser.add_argument('-N', '--NumSeqs', default=5, type=int, help='Number of sequences to simulate')
        parser.add_argument('-min', '--MinLen', default=10, type=int, help='Min len of sequences ')
        parser.add_argument('-max', '--MaxLen', default=100, type=int, help='Max len of sequences ')
        args = parser.parse_args()
        hawkes_raw = neurawkes_master.generate_sequences.main(args)

        # Process the output
        output[user] = {key : {'loc': np.array([VALIDIDS[user][point['type_event']] for point in traj_raw if point['time_since_start'] < 1440])[: -1], 
                            'tim': np.array([point['time_since_start'] for point in traj_raw if point['time_since_start'] < 1440])[: -1], 
                            'sta': np.array([point['time_since_last_event'] for point in traj_raw if point['time_since_start'] < 1440])[1:]}
                        for key, traj_raw in enumerate(hawkes_raw)}

    np.save('./data_hawkes/' + PATH + '/' + MODE + '.npy', output)
    return output

if __name__ == '__main__':
    hawkes('ISP', '0')
    hawkes('ISP', '1')
    hawkes('ISP', '2')

