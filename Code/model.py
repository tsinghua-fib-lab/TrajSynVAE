"""
Created on Sep 15th 18:05:53 2020

Author: Qizhong Zhang

Our VAE model
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from data_prepare import mycollatefunc

# Set the random seeds
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

'''
Embedding Modules: Positional encoding, Sojourn time embedding, location id embedding, user id embedding, POI encoding
'''

# 2 different Fourier encoding
class ABS_TIM_EMB(nn.Module):

    def __init__(self, param):
        super().__init__()
        self.d_model = param.d_model
        self.device = param.device
        self.fourier = param.fourier

    def forward(self, x):
        a = torch.div(torch.arange(0.0, self.d_model), 2, rounding_mode='floor').to(self.device) * 2 / self.d_model
        b = torch.matmul(x.unsqueeze(-1), (2 * np.pi / 1440) * a.unsqueeze(0)) if not self.fourier else \
            torch.matmul(x.unsqueeze(-1), (1e-4 ** a).unsqueeze(0))
        c = torch.zeros_like(b)
        c[:, :, 0::2] = b[:, :, 0::2].sin()
        c[:, :, 1::2] = b[:, :, 1::2].cos()
        return c


# 2 different sojourn time encoding(Embedding ot Linear)
class TIM_DIFF_EMB(nn.Module):

    def __init__(self, param):
        super().__init__()
        self.tim_emb_type = param.tim_emb_type
        self.tim_size = param.tim_size
        self.tim_emb_size = param.tim_emb_size
        if self.tim_emb_type == 'Linear':
            self.emb_tim = nn.Linear(1, self.tim_emb_size, bias=False)
        else:
            self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size, padding_idx=0)

    def forward(self, x):
        if self.tim_emb_type == 'Linear':
            return self.emb_tim(torch.log(x.unsqueeze(-1) + 1e-10))
        return self.emb_tim(x.long())


# Embedding with padding index
class LOC_EMB(nn.Module):

    def __init__(self, param):
        super().__init__()

        self.loc_size = param.loc_size + 1
        self.loc_emb_size = param.loc_emb_size
        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size, padding_idx=0)
        
    def forward(self, x):
        return self.emb_loc(x)


# Embedding with padding index
class USR_EMB(nn.Module):

    def __init__(self, param):
        super().__init__()
        self.USERLIST = np.append(-1, param.USERLIST) + 1
        self.usr_size = param.usr_size + 1
        self.usr_emb_size = param.usr_emb_size
        self.emb_usr = nn.Embedding(self.usr_size, self.usr_emb_size, padding_idx=0)
        self.device = param.device
        
    def forward(self, x):
        usr2id = torch.tensor(np.array([[np.where(self.USERLIST == u.item())[0][0] for u in user] for user in x]))
        return self.emb_usr(usr2id.long().to(self.device))


# 2 different encoding(CDF or Log)
class POI_EMB(nn.Module):

    def __init__(self, param):
        super().__init__()
        self.POI = np.concatenate((np.zeros((1, param.POI.shape[1])), param.POI), axis=0)
        if param.cdfpoi:
            self.CDF = [np.append(0, (np.cumsum(np.bincount(poi.astype(int))) / poi.shape[0]))[:-1] for poi in self.POI.T]
            self.POI = np.array([[self.CDF[i][int(j)] for j in self.POI[:, i]] for i in range(self.POI.shape[1])]).T
        else:
            self.POI = np.log((self.POI + 1))
        self.device = param.device

    def forward(self, x):
        return torch.tensor(np.array([self.POI[np.ix_(locs.cpu())] for locs in x])).double().to(self.device)

'''
Initialization functions for RNNs(Xavier for LSTM, Orthogonal for GRU) and Linear Layers(KaiMing)
'''

def initialize_rnn(model, type='LSTM'):
    for name, param in model.named_parameters():
        if 'weight_ih' in name:
            torch.nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            if type == 'LSTM':
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.orthogonal_(param)
        elif 'bias' in name:
            torch.nn.init.constant_(param, 0.0)


def initialize_linear_layer(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.kaiming_uniform_(layer.weight,  a = 0.01)

'''
Basic Encoder Model
'''

class ENCODER(nn.Module):

    def __init__(self, param):
        super(ENCODER, self).__init__()

        # Ablation
        self.device = param.device
        self.d_model = param.d_model if not param.pos_emb_ban else 0
        self.poi_size = param.poi_size if not param.poi_emb_ban else 0

        # Encoder RNN
        self.encoder_rnn_input_size = param.loc_emb_size + param.tim_emb_size + param.usr_emb_size + self.d_model + self.poi_size
        self.encoder_rnn_hidden_size = param.encoder_rnn_hidden_size
        self.rnn_type = param.rnn_type
        self.rnn_layers = param.rnn_layers
        self.rnn_bidirectional = param.rnn_bidirectional

        if self.rnn_type == 'LSTM':
            self.encoder_rnn = nn.LSTM(self.encoder_rnn_input_size, self.encoder_rnn_hidden_size, num_layers=self.rnn_layers, batch_first=True, bidirectional=self.rnn_bidirectional)
        else:
            self.encoder_rnn = nn.GRU(self.encoder_rnn_input_size, self.encoder_rnn_hidden_size, num_layers=self.rnn_layers, batch_first=True, bidirectional=self.rnn_bidirectional)
        
        # Regularization
        self.layernorm = param.layernorm
        self.layer_norm = nn.LayerNorm(self.encoder_rnn_hidden_size * (1 + self.rnn_bidirectional))
        
        # Gauss distribution for latent variable
        self.z_hidden_size_mean = param.z_hidden_size_mean
        self.z_hidden_size_std = param.z_hidden_size_std
        self.latent_size = param.latent_size

        self.mean_l1 = nn.Linear(self.encoder_rnn_hidden_size * (1 + self.rnn_bidirectional), self.z_hidden_size_mean)
        self.mean_l2 = nn.Linear(self.z_hidden_size_mean, self.latent_size)
        self.std_l1 = nn.Linear(self.encoder_rnn_hidden_size * (1 + self.rnn_bidirectional), self.z_hidden_size_std)
        self.std_l2 = nn.Linear(self.z_hidden_size_std, self.latent_size)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                initialize_linear_layer(m)
            if isinstance(m, nn.LSTM):
                initialize_rnn(m, self.rnn_type)
            if isinstance(m, nn.GRU):
                initialize_rnn(m, self.rnn_type)

    # Forward function
    def forward(self, x_emb, lengths = None): 

        # Encoder RNN
        lengths = lengths if lengths is not None else [x_emb.shape[1]] * x_emb.shape[0]
        packed_input = pack_padded_sequence(input=x_emb, lengths=lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.encoder_rnn(packed_input)
        hidden0, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=x_emb.shape[1])    

        # Regularization
        hidden = self.layer_norm(hidden0) if self.layernorm else hidden0 

        # param for Gauss distribution
        mean = self.mean_l2(F.leaky_relu(self.mean_l1(hidden)))
        std = self.std_l2(F.leaky_relu(self.std_l1(hidden)))

        return mean, std 

'''
Basic Decoder Model
'''

class DECODER(nn.Module):

    def __init__(self, param):
        super(DECODER, self).__init__()

        # Ablation
        self.device = param.device
        self.d_model = param.d_model if not param.feedback_ban else 0
        self.poi_ban = param.poi_ban

        # Decoder RNN
        self.decoder_rnn_input_size = param.latent_size + param.usr_emb_size + self.d_model
        self.decoder_rnn_hidden_size = param.decoder_rnn_hidden_size
        self.rnn_type = param.rnn_type
        self.rnn_layers = param.rnn_layers
        self.rnn_bidirectional = param.rnn_bidirectional
        self.dual_rnn = param.dual_rnn

        if self.rnn_type == 'LSTM':
            self.decoder_rnn = nn.LSTM(self.decoder_rnn_input_size, self.decoder_rnn_hidden_size, num_layers=self.rnn_layers, batch_first=True, bidirectional=self.rnn_bidirectional)
            if self.dual_rnn == True:
                self.decoder_rnn1 = nn.LSTM(self.decoder_rnn_input_size, self.decoder_rnn_hidden_size, num_layers=self.rnn_layers, batch_first=True, bidirectional=self.rnn_bidirectional)
        else:
            self.decoder_rnn = nn.GRU(self.decoder_rnn_input_size, self.decoder_rnn_hidden_size, num_layers=self.rnn_layers, batch_first=True, bidirectional=self.rnn_bidirectional)
            if self.dual_rnn == True:
                self.decoder_rnn1 = nn.GRU(self.decoder_rnn_input_size, self.decoder_rnn_hidden_size, num_layers=self.rnn_layers, batch_first=True, bidirectional=self.rnn_bidirectional)

        # Regularization
        self.layernorm = param.layernorm
        self.layer_norm = nn.LayerNorm(self.decoder_rnn_hidden_size * (1 + self.rnn_bidirectional))
        self.dropout = nn.Dropout(param.dropout)

        # Location decoder
        self.loc_hidden_size1 = param.loc_hidden_size1
        self.loc_hidden_size2 = param.loc_hidden_size2
        self.loc_size = param.loc_size + 1
        self.poi_size = param.poi_size

        self.loc_l1 = nn.Linear(self.decoder_rnn_hidden_size * (1 + self.rnn_bidirectional), self.loc_hidden_size1)
        self.loc_l2 = nn.Linear(self.loc_hidden_size1, self.loc_hidden_size2)
        self.loc_l3 = nn.Linear(self.loc_hidden_size2, self.loc_size)

        if self.poi_size and not self.poi_ban:
            self.loc_l4 = nn.Linear(self.loc_hidden_size2, self.poi_size)
            self.loc_l5 = nn.Linear(self.poi_size, 1, bias=None)

            self.POI = np.concatenate((np.zeros((1, param.POI.shape[1])), param.POI), axis=0)
            if param.cdfpoi:
                self.CDF = [np.append(0, (np.cumsum(np.bincount(poi.astype(int))) / poi.shape[0]))[:-1] for poi in self.POI.T]
                self.POI = np.array([[self.CDF[i][int(j)] for j in self.POI[:, i]] for i in range(self.POI.shape[1])]).T
            else:
                self.POI = np.log((self.POI + 1))

            if param.poi_weight_dynamic:
                self.poi_weight = nn.Parameter(torch.tensor(-np.log(1 / param.poi_weight - 1)))
            else:
                self.poi_weight = torch.tensor(-np.log(1 / param.poi_weight - 1))

        # Sojourn time decoder
        self.tim_hidden_size1 = param.tim_hidden_size1
        self.tim_hidden_size2 = param.tim_hidden_size2

        self.tim_l1 = nn.Linear(self.decoder_rnn_hidden_size * (1 + self.rnn_bidirectional), self.tim_hidden_size1)
        self.tim_l2 = nn.Linear(self.tim_hidden_size1, self.tim_hidden_size2)
        self.tim_l3 = nn.Linear(self.tim_hidden_size2, 1)
        self.time_initial = param.time_initial

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                initialize_linear_layer(m)
            if isinstance(m, nn.LSTM):
                initialize_rnn(m, self.rnn_type)
            if isinstance(m, nn.GRU):
                initialize_rnn(m, self.rnn_type)

    def forward(self, z, lengths = None):

        # Decoder RNN
        lengths = lengths if lengths is not None else [z.shape[1]] * z.shape[0]
        packed_input = pack_padded_sequence(input=z, lengths=lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.decoder_rnn(packed_input)
        hidden0, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=z.shape[1])
        hidden = self.layer_norm(hidden0) if self.layernorm else hidden0 
        if self.dual_rnn == True:
            packed_output1, _ = self.decoder_rnn(packed_input)
            hiddenT, _ = pad_packed_sequence(packed_output1, batch_first=True, total_length=z.shape[1])
            hidden1 = self.layer_norm(hiddenT) if self.layernorm else hiddenT 

        # Location decoder 
        lout2 = F.leaky_relu(self.loc_l2(F.leaky_relu(self.loc_l1(self.dropout(hidden)))))
        if self.poi_size and not self.poi_ban:
            lout_1 = F.log_softmax(self.loc_l3(lout2), dim=2)
            POI = torch.tensor(self.POI).to(self.device)
            # B*L*S -> B*L*P -> B*L*1*P -> B*L*N*P -> B*L*N*1 -> B*L*N
            lout_2 = F.log_softmax(self.loc_l5(F.leaky_relu(self.loc_l4(lout2)).unsqueeze(-2) * POI).squeeze(-1), dim=2)
            lout = torch.logaddexp(torch.log(1 - torch.sigmoid(self.poi_weight)) + lout_1, F.logsigmoid(self.poi_weight) + lout_2)
        else:
            lout = F.log_softmax(self.loc_l3(lout2), dim=2)

        # Waiting time decoder
        tout = self.tim_l3(F.leaky_relu(self.tim_l2(F.leaky_relu(self.tim_l1(hidden))))).squeeze(-1)
        if self.dual_rnn == True:
            tout = self.tim_l3(F.leaky_relu(self.tim_l2(F.leaky_relu(self.tim_l1(hidden1))))).squeeze(-1)

        return lout, tout - np.log(self.time_initial)

'''
VAE MODEL: Inference and Generation
'''

class VAE(nn.Module):

    def __init__(self, param):

        super().__init__()

        # Ablation
        self.tim_size = param.tim_size
        self.poi_size = param.poi_size
        self.loc_size = param.loc_size + 1
        self.latent_size = param.latent_size
        self.pos_emb_ban = param.pos_emb_ban
        self.poi_emb_ban = param.poi_emb_ban
        self.feedback_ban = param.feedback_ban

        # Embedding
        self.emb_loc = LOC_EMB(param)
        self.emb_tim = TIM_DIFF_EMB(param)
        self.emb_usr = USR_EMB(param)
        self.emb_pos = ABS_TIM_EMB(param) if not self.pos_emb_ban else None
        self.emb_poi = POI_EMB(param) if ((self.poi_emb_ban == False) and (self.poi_size > 0)) else None

        self.emb_pos0 = ABS_TIM_EMB(param) if not self.feedback_ban else None
        self.emb_usr0 = USR_EMB(param)
        # Sharing weights
        for para_0, para in zip(self.emb_usr0.parameters(), self.emb_usr.parameters()):
            para_0.data.copy_(para.data)  # initialize
            para_0.requires_grad = False  # not updated by gradient

        # Modules
        self.encoder = ENCODER(param)
        self.decoder = DECODER(param)

        # Generation
        self.USERLIST = param.USERLIST
        self.infer_maxlast = param.infer_maxlast
        self.first_sample = param.first_sample
        self.ntrajs = param.ntrajs

        # KL-annealing
        self.max_beta = param.max_beta
        self.cycle = param.cycle
        # Inference
        self.learning_rate = param.learning_rate
        self.L2 = param.L2
        self.step_size = param.step_size
        self.gamma = param.gamma
        self.epoches = param.epoches
        self.batchsize = param.batchsize

        # Tuning
        self.tune = param.tune
        self.tuned = 0

        # Necessary attributes preparation
        self.user_indicator = np.ones((param.USERLIST.shape[0]+1, param.loc_size+1))
        self.loc_weights = np.ones((param.USERLIST.shape[0]+1, param.loc_size+1))
        self.initial_prob = np.ones(self.tim_size // 10)
        self.save_path = param.save_path
        self.device = param.device
        self.loc_initial = param.loc_initial

        # Unbalanced learning
        self.tim_only = param.tim_only
        if self.tim_only:
            timdecoder = ['tim_l1', 'tim_l2', 'tim_l3']
            for name, para in self.named_parameters():
                modulename, layername = name.split('.')[0], name.split('.')[1]
                if (layername not in timdecoder) and (modulename != 'emb_usr0'):
                    para.requires_grad = False

        self.loc_only = param.loc_only
        if self.loc_only:
            locdecoder = ['loc_l1', 'loc_l2', 'loc_l3', 'loc_l4', 'loc_l5']
            for name, para in self.named_parameters():
                modulename, layername = name.split('.')[0], name.split('.')[1]
                if (layername not in locdecoder) and (modulename != 'emb_usr0'):
                    para.requires_grad = False


    def forward(self, inseq):

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._embedding_update()

        # Embedding of location, waiting time and user ID
        loc_emb = self.emb_loc(inseq['loc'] + 1)
        tim_emb = self.emb_tim(inseq['tim'])
        usr_emb = self.emb_usr(inseq['usr'] + 1)
        pos_emb = self.emb_pos(inseq['pos']) if not self.pos_emb_ban else torch.tensor([]).to(self.device)
        poi_emb = self.emb_poi(inseq['loc'] + 1) if ((self.poi_emb_ban == False) and (self.poi_size > 0)) else torch.tensor([]).to(self.device)
        x_emb = torch.cat((loc_emb, poi_emb, tim_emb, usr_emb, pos_emb), -1)

        # Encoder
        mean, logstd = self.encoder(x_emb, inseq['lengths'])

        # Sampling
        z = torch.randn_like(logstd).to(self.device)
        z = z * torch.exp(logstd) + mean

        # Embedding
        usr_emb0 = self.emb_usr0(inseq['usr'] + 1)
        pos_emb0 = self.emb_pos0(inseq['pos']) if not self.feedback_ban else torch.tensor([]).to(self.device)
        z = torch.cat((z, usr_emb0, pos_emb0), -1)

        # Decoder
        lout, tout = self.decoder(z, inseq['lengths'])
        lout = self.locprob_filter(lout, inseq['usr'])

        return mean, logstd, lout, tout

    # Only the locations in the training set of one user will be involved in corresponding inference and generation for this user
    def locprob_filter(self, lout, usr, training = True):
        user_encoding = np.array([[np.where(np.append(-1, self.USERLIST) == u.item())[0][0] for u in user] for user in usr]).astype(int)
        weights = torch.tensor(np.array([self.user_indicator[np.ix_(usrs)] for usrs in user_encoding])).to(self.device)
        l0 = lout - torch.log(weights) if training else lout.masked_fill(weights != 1, float('-inf'))
        l1 = l0 - torch.logsumexp(l0, dim=2, keepdim=True)
        return l1

    # Update Embedding
    @torch.no_grad()
    def _embedding_update(self):
        for param_0, param in zip(self.emb_usr0.parameters(), self.emb_usr.parameters()):
            param_0.data = param.data

    # ELBO Loss with KL-annealing and dynamic weight adjestment
    def loss(self, mean, logstd, lout, tout, inseq, weight = 0.5, beta = 1):

        mask = pad_sequence([torch.tensor([1] * int(l)) for l in inseq['lengths']], batch_first=True).to(self.device)
        percentage = torch.sum(mask) / (mask.shape[0] * mask.shape[1])
        KL = torch.mean(0.5 * torch.sum(torch.exp(2 * logstd) - 1 - 2 * logstd + mean.pow(2), dim=-1) * mask) / percentage

        LL_T = -torch.mean((tout - torch.exp(tout) * inseq['tim'])  * mask) / percentage
        # LL_T = torch.mean((tout + torch.log(torch.where(inseq['tim'] < 1, 1.0, inseq['tim']))).pow(2)  * mask) / percentage
        LL_T0 = -torch.mean((torch.log1p(-torch.exp(-torch.exp(tout))) - torch.exp(tout) * inseq['tim']) * mask) / percentage

        LL_L = torch.mean(nn.NLLLoss(reduction='none')(lout.swapaxes(1,2), inseq['loc'] + 1) * mask) / percentage
        user_encoding = np.array([[np.where(np.append(-1, self.USERLIST) == u.item())[0][0] for u in user] for user in inseq['usr']]).astype(int)
        indicator = np.array([self.user_indicator[np.ix_(usrs)] for usrs in user_encoding])
        indicator = torch.tensor(np.where(indicator != 1, 0, 1)).to(self.device)
        LL_L0 = nn.NLLLoss(reduction='none')((lout * indicator).swapaxes(1,2), inseq['loc'] + 1) * mask
        LL_L0 = torch.mean(LL_L0) / (torch.sum(LL_L0 != 0) / torch.cumprod(torch.tensor(LL_L0.shape), 0)[-1])

        LOSS = beta * KL + (1 - weight) * LL_T + weight * LL_L

        return KL, LL_L, LL_T, LOSS, LL_L0, LL_T0

    # Sample the sojourn time and location time 
    def sample(self, lout, tout, last_loc, ntrajs = 7): 

        Lambda = torch.exp(tout[:, -1]).unsqueeze(1).cpu().detach().numpy()
        def truncated_exponential_samples(size, lower, upper, rate):
            uniform_samples = np.random.rand(*size)
            rate0 = np.where(rate < 1e-10, 1e-10, rate)
            truncated_samples = -np.log(1 - uniform_samples * (1 - np.exp(-rate0 * (upper - lower)))) / rate0 + lower
            return truncated_samples
        t = np.squeeze(truncated_exponential_samples((ntrajs, 1), 1, self.infer_maxlast, Lambda))

        prob = torch.exp(lout[:, -1, :]).squeeze(1).cpu().detach().numpy()
        for id, loc in enumerate(last_loc):
            prob[id, list(set([0, loc]))] = 0
        l = np.array([np.random.choice(list(range(self.loc_size)), p=P/sum(P)) for P in prob])

        return l, t

    # Generating for one user
    @torch.no_grad()
    def inference(self, user, ntrajs = 7):

        # Decoder & Generate
        with torch.no_grad():

            # Without feedback
            if self.feedback_ban:

                # Data preparation
                z = torch.randn(ntrajs, 144, self.latent_size).to(self.device)
                usr = (user + 1) * torch.ones((ntrajs, 144), dtype=torch.long).to(self.device)
                last_location = np.zeros(ntrajs).astype(int)

                # Sample the first data point
                X = {'loc': [], 'tim': [], 'sta': []}
                lout, tout = self.decoder(torch.cat((z, self.emb_usr0(usr)), -1))
                lout = self.locprob_filter(lout, usr-1)

                # Generation
                for i in range(1, 145):
                    l, t = self.sample(lout[:, :i, :], tout[:, :i], last_location)
                    X['loc'].append(l)
                    X['sta'].append(t)
                    last_location = l.astype(int)

                    if np.min(np.sum(np.array(X['sta']), axis=0)) >= self.infer_maxlast:
                        break
                
                # Get standard output
                output = {}
                for i in range(ntrajs):
                    id = np.where(np.cumsum(np.array(X['sta'])[:, i]) >= self.infer_maxlast)[0][0]
                    output[i] = {'loc': np.array(X['loc'])[1:(1+id), i] - 1, 
                                 'tim': np.cumsum(np.array(X['sta'])[:id, i]), 
                                 'sta': np.array(X['sta'])[1:(1+id), i]}
                    
                return output

            # Generating with Feedback

            # sample starting point
            z = torch.randn(ntrajs, 1000, self.latent_size).to(self.device)
            usr = (user + 1) * torch.ones((ntrajs, 1000), dtype=torch.long).to(self.device)

            if self.first_sample =='New':
                t = np.random.choice(np.arange(self.tim_size // 10), size=self.ntrajs, p=self.initial_prob) * 10 + 5
                last_location = np.zeros(ntrajs).astype(int)
            else:
                usr_emb0 = self.emb_usr0(usr[:,0].unsqueeze(1))
                pos_emb0 = self.emb_pos0(torch.ones((ntrajs,1), dtype=torch.double).to(self.device) * 10)
                lout1, tout1 = self.decoder(torch.cat((z[:,0,:].unsqueeze(1), usr_emb0, pos_emb0), -1))
                lout1 = self.locprob_filter(lout1, usr[:,0].unsqueeze(1)-1)
                l, t = self.sample(lout1, tout1, last_location, ntrajs = ntrajs)
                last_location = l.astype(int)

            # Generating
            X = {'loc': [], 'tim': [t], 'sta': []}
            for i in range(1, 1000):

                time = torch.tensor(np.array(X['tim']).T, dtype=torch.double).to(self.device)
                usr_emb0 = self.emb_usr0(usr[:, :i])
                pos_emb0 = self.emb_pos0(time)
                louti, touti = self.decoder(torch.cat((z[:, :i, :], usr_emb0, pos_emb0), -1))
                louti = self.locprob_filter(louti, usr[:, :i]-1)

                l, t = self.sample(louti, touti, last_location)
                X['tim'].append(np.array(X['tim'])[-1] + t)
                X['loc'].append(l)
                X['sta'].append(t)
                last_location = np.array(X['loc'])[-1].astype(int)

                if np.min(np.array(X['tim'])[-1]) >= self.infer_maxlast:
                    break  

            # Get standard output
            output = {}
            for i in range(ntrajs):
                id = np.where(np.array(X['tim'])[:, i] >= self.infer_maxlast)[0][0]
                output[i] = {'loc': np.array(X['loc'])[:id, i] - 1,
                             'tim': np.array(X['tim'])[:id, i], 
                             'sta': np.array(X['sta'])[:id, i]}
                
            return output

    def load(self, cp):
        self.load_state_dict(torch.load(cp))
        print("Load model from %s" % cp )

    def save(self, cp):
        torch.save(self.state_dict(), cp)
        print("Model saved as %s" % cp)

    # Generating for all users
    @torch.no_grad()
    def test_data_prepare(self, data, load_checkpoint = None):
        
        self.eval()
        if load_checkpoint != None:
            self.load(load_checkpoint)

        with torch.no_grad():
            output_sequence = {}
            gen_bar = tqdm(data.REFORM['test'])
            for user in gen_bar:
                gen_bar.set_description("Generating trajectories for user: {}".format(user))
                output_sequence[user] = self.inference(user, self.ntrajs)                

            data.GENDATA.append(output_sequence)
            np.save(self.save_path + 'data/generated_' + str(self.tuned) + '.npy', output_sequence)

        self.tuned += 1

    # Count for all locations that appear in the historical trajectories for each user
    def location_constraints(self, data):
        user_indicator = np.ones((self.USERLIST.shape[0]+1, self.loc_size))
        weights = np.ones((self.USERLIST.shape[0]+1, self.loc_size))
        for userid, data_user in data.items():
            count = np.bincount(np.concatenate([traj['loc']+1 for traj in data_user.values()]))
            count = np.append(count, np.zeros(self.loc_size - count.shape[0]))
            weights[np.where(self.USERLIST == userid)[0][0]+1] = count
            user_indicator[(np.where(self.USERLIST == userid)[0][0]+1, count == 0)] = self.loc_initial
        weights[weights > 0] = np.log(weights[weights > 0]) + 1
        return user_indicator, weights
    
    # Count for starting timestamps of all trajectories in training set for generation
    def inference_initial(self, data):
        timestamps = np.concatenate([[traj['tim'][0] % 1440 for traj in trajs.values()] for trajs in data.dataset.REFORM['train'].values()])
        freqs = np.histogram(timestamps, bins = self.tim_size // 10, range=(0, self.tim_size))[0]
        return freqs / freqs.sum()
        
    # Training
    def Train(self, trainset, validset):

        self.train()

        # Set optimizer and scheduler
        print('Start Training')
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,  weight_decay=self.L2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

        # Necessary attributes
        self.user_indicator, self.loc_weights = self.location_constraints(trainset.dataset.REFORM['train'])
        self.initial_prob = self.inference_initial(trainset)

        # KL-annealing
        def cyclical_KL_annealing(step, cycle, R=0.5, M=1):
            step0 = step % cycle
            M = M * step0 / (cycle * R) if step0 < cycle * R else M
            return M, step+1

        # Train
        loss_record, valid_record, step, weight = {}, {}, 0, 0.5 
        for epoch in range(1, self.epoches + 1):
            optimizer.zero_grad()
            loss_record[epoch] = {"LOSS": [], "KL": [], "LL_T": [], "LL_L": [], "weight" : [], "LL_L0": [], "LL_T0": []}

            mydataloader = DataLoader(trainset, batch_size=self.batchsize, shuffle=True, num_workers=0, collate_fn=mycollatefunc)
            train_bar = tqdm(enumerate(mydataloader))
            for idx, bat in train_bar:

                # Input
                loc = bat['loc'].clone().detach().long().to(self.device)
                tim = bat['sta'].clone().detach().double().to(self.device)
                usr = bat['usr'].clone().detach().long().to(self.device)
                pos = bat['tim'].clone().detach().double().to(self.device)
                inseq = {'usr': usr, 'loc': loc, 'tim': tim, 'pos': pos, 'lengths': bat['lengths']}

                # Output
                mean, std, lout, tout = self.forward(inseq)

                # Loss
                beta, step = cyclical_KL_annealing(step, self.cycle, M=self.max_beta)
                if self.tim_only:
                    beta, weight = 0, 0
                if self.loc_only:
                    beta, weight = 0, 1
                KL, LL_L, LL_T, LOSS, LL_L0, LL_T0 = self.loss(mean, std, lout, tout, inseq, weight=weight, beta=beta)

                # Backward
                LOSS.backward()
                optimizer.step()

                # Loss record
                loss_record[epoch]['KL'].append(KL.item())
                loss_record[epoch]['LL_L'].append(LL_L.item())
                loss_record[epoch]['LL_T'].append(LL_T.item())
                loss_record[epoch]['LOSS'].append(LOSS.item())
                loss_record[epoch]['LL_L0'].append(LL_L0.item())
                loss_record[epoch]['LL_T0'].append(LL_T0.item())

                train_bar.set_description('Training epoch: {}, weight_L: {:.5f}, Index: {}, KL: {:.5f}, LL_L: {:.5f}, LL_L0: {:.5f}, LL_T: {:.5f}, LL_T0: {:.5f}, ELBO: {:.5f}'.format\
                                           (epoch, weight, idx, np.mean(loss_record[epoch]['KL']),
                                            np.mean(loss_record[epoch]['LL_L']),
                                            np.mean(loss_record[epoch]['LL_L0']),
                                            np.mean(loss_record[epoch]['LL_T']),
                                            np.mean(loss_record[epoch]['LL_T0']),
                                            np.mean(loss_record[epoch]['LOSS'])))

            # Validation
            if epoch % 1 == 0:

                self.eval()

                with torch.no_grad():

                    valid_record[epoch] = {"LOSS": [], "KL": [], "LL_T": [], "LL_L": [], "LL_L0": [], "LL_T0": [], "weight" : []}
                    mydataloader = DataLoader(validset, batch_size=self.batchsize, shuffle=False, num_workers=0, collate_fn=mycollatefunc)
                    valid_bar = tqdm(enumerate(mydataloader))
                    for idx, bat in valid_bar:

                        # Input
                        loc = bat['loc'].clone().detach().long().to(self.device)
                        tim = bat['sta'].clone().detach().double().to(self.device)
                        usr = bat['usr'].clone().detach().long().to(self.device)
                        pos = bat['tim'].clone().detach().double().to(self.device)
                        inseq = {'usr': usr, 'loc': loc, 'tim': tim, 'pos': pos, 'lengths': bat['lengths']}

                        # Output
                        mean, std, lout, tout = self.forward(inseq)

                        # Loss
                        KL, LL_L, LL_T, LOSS, LL_L0, LL_T0 = self.loss(mean, std, lout, tout, inseq, weight=weight, beta=beta)

                        # Loss record
                        valid_record[epoch]['KL'].append(KL.item())
                        valid_record[epoch]['LL_L'].append(LL_L.item())
                        valid_record[epoch]['LL_T'].append(LL_T.item())
                        valid_record[epoch]['LOSS'].append(LOSS.item())
                        valid_record[epoch]['LL_L0'].append(LL_L0.item())
                        valid_record[epoch]['LL_T0'].append(LL_T0.item())

                        valid_bar.set_description('Validation epoch: {}, Index: {}, KL: {:.5f}, LL_L: {:.5f}, LL_L0: {:.5f}, LL_T: {:.5f}, LL_T0: {:.5f}, ELBO: {:.5f}'.format\
                                            (epoch, idx, np.mean(valid_record[epoch]['KL']),
                                                np.mean(valid_record[epoch]['LL_L']),
                                                np.mean(valid_record[epoch]['LL_L0']),
                                                np.mean(valid_record[epoch]['LL_T']),
                                                np.mean(valid_record[epoch]['LL_T0']),
                                                np.mean(valid_record[epoch]['LOSS'])))
                        
                self.train()

            # Reduce the learning rate and adjusting the weight
            weight = 1 / (1 + np.var(loss_record[epoch]['LL_L']) / np.var(loss_record[epoch]['LL_T']))
            loss_record[epoch]['weight'].append(weight)
            scheduler.step()

            # Checkpoint
            if epoch % 10 == 0:
                if self.tune == True:
                    self.save(self.save_path + 'data/Model' + str(epoch//10) + '.pth')
                else:
                    self.save(self.save_path + 'data/Model.pth')

        self.save(self.save_path + 'data/Model.pth')

        return loss_record, valid_record

    # Testing
    def test(self, testset):

        self.eval()

        with torch.no_grad():

            # Testing
            test_record = {"LOSS": [], "KL": [], "LL_T": [], "LL_L": []}
            mydataloader = DataLoader(testset, batch_size=self.batchsize, shuffle=False, num_workers=0, collate_fn=mycollatefunc)
            test_bar = tqdm(enumerate(mydataloader))
            for idx, bat in test_bar:

                # Input
                loc = bat['loc'].clone().detach().long().to(self.device)
                tim = bat['sta'].clone().detach().double().to(self.device)
                usr = bat['usr'].clone().detach().long().to(self.device)
                pos = bat['tim'].clone().detach().double().to(self.device)
                inseq = {'usr': usr, 'loc': loc, 'tim': tim, 'pos': pos, 'lengths': bat['lengths']}

                # Output
                mean, std, lout, tout = self.forward(inseq)

                # Loss
                KL, LL_L, LL_T, LOSS, LL_L0, LL_T0 = self.loss(mean, std, lout, tout, inseq)

                # Loss record
                test_record['KL'].append(KL.item())
                test_record['LL_L'].append(LL_L.item())
                test_record['LL_T'].append(LL_T.item())
                test_record['LOSS'].append(LOSS.item())

                test_bar.set_description('Testing index: {}, KL: {:.5f}, LL_L: {:.5f}, LL_T: {:.5f}, ELBO: {:.5f}'.format\
                                            (idx, np.mean(test_record['KL']),
                                                np.mean(test_record['LL_L']),
                                                np.mean(test_record['LL_T']),
                                                np.mean(test_record['LOSS'])))
                
        # Genration
        if self.tune == True:
            for i in range(1, 1 + self.epoches // 10):
                self.test_data_prepare(testset.dataset, load_checkpoint = self.save_path + 'data/Model' + str(i) + '.pth')
        else:
            self.test_data_prepare(testset.dataset)

        return test_record

    # Plot the losses
    def loss_plot(self, loss_record, valid_record, test_record):

        KL = [np.mean(loss_record[epoch]['KL']) for epoch in loss_record]
        LL_L = [np.mean(loss_record[epoch]['LL_L']) for epoch in loss_record]
        LL_T = [np.mean(loss_record[epoch]['LL_T']) for epoch in loss_record]
        LOSS = [np.mean(loss_record[epoch]['LOSS']) for epoch in loss_record]

        valid_KL = [np.mean(valid_record[epoch]['KL']) for epoch in valid_record]
        valid_LL_L = [np.mean(valid_record[epoch]['LL_L0']) for epoch in valid_record]
        valid_LL_T = [np.mean(valid_record[epoch]['LL_T']) for epoch in valid_record]
        valid_LOSS = [np.mean(valid_record[epoch]['LOSS']) for epoch in valid_record]

        x = np.array([epoch for epoch in loss_record])
        y = np.array([epoch for epoch in valid_record])
    
        plt.figure()

        plt.subplot(221)
        ln1, = plt.plot(x, KL, color='red', linewidth=2.0, linestyle='-')
        ln2, = plt.plot(y, valid_KL, color='blue', linewidth=2.0, linestyle='-')
        plt.title('KL, Test = ' + str(np.around(np.mean(test_record['KL']), decimals=3)))
        plt.xlabel('Epoches')
        plt.ylabel('KLLoss')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(handles=[ln1, ln2], labels=['Train', 'Valid'])

        plt.subplot(222)
        ln1, = plt.plot(x, LL_L, color='red', linewidth=2.0, linestyle='-')
        ln2, = plt.plot(y, valid_LL_L, color='blue', linewidth=2.0, linestyle='-')
        plt.title('LL_L, Test = ' + str(np.around(np.mean(test_record['LL_L']), decimals=3)))
        plt.xlabel('Epoches')
        plt.ylabel('LL_LLoss')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(handles=[ln1, ln2], labels=['Train', 'Valid'])

        plt.subplot(223)
        ln1, = plt.plot(x, LL_T, color='red', linewidth=2.0, linestyle='-')
        ln2, = plt.plot(y, valid_LL_T, color='blue', linewidth=2.0, linestyle='-')
        plt.title('LL_T, Test = ' + str(np.around(np.mean(test_record['LL_T']), decimals=3)))
        plt.xlabel('Epoches')
        plt.ylabel('LL_TLoss')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(handles=[ln1, ln2], labels=['Train', 'Valid'])

        plt.subplot(224)
        ln1, = plt.plot(x, LOSS, color='red', linewidth=2.0, linestyle='-')
        ln2, = plt.plot(y, valid_LOSS, color='blue', linewidth=2.0, linestyle='-')
        plt.title('ELBO, Test = ' + str(np.around(np.mean(test_record['LOSS']), decimals=3)))
        plt.xlabel('Epoches')
        plt.ylabel('ELBOLoss')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(handles=[ln1, ln2], labels=['Train', 'Valid'])

        plt.tight_layout()
        plt.savefig(self.save_path + 'plots' +  '/Loss_plot.png')

    # CALL THIS FUCNTION TO DO INFERENCE AND GENERATION
    def run(self, trainset, testset):
        loss_record, valid_record = self.Train(trainset, testset)
        test_record = self.test(testset)
        self.loss_plot(loss_record, valid_record, test_record)
        