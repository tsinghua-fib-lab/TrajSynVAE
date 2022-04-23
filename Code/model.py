# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
Created on Sep 15th 18:05:53 2020

Our model
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from data_prepare import mycollatefunc


# Sets the seed for generating random numbers
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

class ABS_TIM_EMB(nn.Module):

    def __init__(self, param):
        super().__init__()
        self.d_model = param.d_model
        self.device = param.device
        self.fourier = param.fourier

    def forward(self, x):
        a = torch.div(torch.arange(0.0, self.d_model).to(self.device), 2, rounding_mode='floor') * 2
        b = torch.transpose(torch.matmul((2 * np.pi / 10080) * (a / self.d_model).unsqueeze(1), x.unsqueeze(1)), 1, 2) if not self.fourier else \
            torch.transpose(torch.matmul((1e-4 ** (a / self.d_model)).unsqueeze(1), x.unsqueeze(1)), 1, 2)
        c = torch.zeros_like(b)
        c[:, 0::2] = b[:, 0::2].sin()
        c[:, 1::2] = b[:, 1::2].cos()
        return c


class TIM_DIFF_EMB(nn.Module):

    def __init__(self, param):
        super().__init__()

        self.tim_size = param.tim_size
        self.tim_emb_size = param.tim_emb_size
        self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)

    def forward(self, x):
        return self.emb_tim(x)


class LOC_EMB(nn.Module):

    def __init__(self, param):
        super().__init__()

        self.loc_size = param.loc_size 
        self.loc_emb_size = param.loc_emb_size
        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        
    def forward(self, x):
        return self.emb_loc(x)


class USR_EMB(nn.Module):

    def __init__(self, param):
        super().__init__()
        self.USERLIST = np.append(-1, param.USERLIST)
        self.usr_size = param.usr_size + 1
        self.usr_emb_size = param.usr_emb_size
        self.emb_usr = nn.Embedding(self.usr_size, self.usr_emb_size)
        self.device = param.device
        
    def forward(self, x):
        usr2id = torch.tensor(np.array([[np.where(self.USERLIST == u.item())[0][0] for u in x[user]] for user in range(len(x))])).long().to(self.device)
        return self.emb_usr(usr2id)


class POI_EMB(nn.Module):

    def __init__(self, param):
        super().__init__()
        self.POI = param.POI
        self.device = param.device

    def forward(self, x):
        return torch.tensor(np.array([self.POI[np.ix_(locs.cpu())] for locs in x]), dtype=torch.double).to(self.device)


class ENCODER(nn.Module):

    def __init__(self, param, emb):
        super(ENCODER, self).__init__()

        # Embedding
        self.emb_loc = emb[0]
        self.emb_tim = emb[1]
        self.emb_usr = emb[2]
        self.d_model = param.d_model if not param.pos_emb_ban else 0
        self.poi_size = param.poi_size if not param.poi_emb_ban else 0
        self.emb_pos = emb[3] if self.d_model else None
        self.emb_poi = emb[4] if self.poi_size else None

        # Encoder RNN
        self.encoder_rnn_input_size = param.loc_emb_size + param.tim_emb_size + param.usr_emb_size + self.d_model + self.poi_size
        self.encoder_rnn_hidden_size = param.encoder_rnn_hidden_size
        self.encoder_rnn = nn.LSTM(self.encoder_rnn_input_size, self.encoder_rnn_hidden_size, 1, batch_first=True)

        # Gauss distribution for latent variable
        self.z_hidden_size_mean = param.z_hidden_size_mean
        self.z_hidden_size_std = param.z_hidden_size_std
        self.latent_size = param.latent_size
        self.mean_l1 = nn.Linear(self.encoder_rnn_hidden_size, self.z_hidden_size_mean)
        self.mean_l2 = nn.Linear(self.z_hidden_size_mean, self.latent_size)
        self.std_l1 = nn.Linear(self.encoder_rnn_hidden_size, self.z_hidden_size_std)
        self.std_l2 = nn.Linear(self.z_hidden_size_std, self.latent_size)

        # Dropout
        self.dropout = nn.Dropout(p=param.dropout)

        self.device = param.device

    # Forward function
    def forward(self, usr, loc, tim, pos):

        # Embedding of location, waiting time and user ID
        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        usr_emb = self.emb_usr(usr)
        if self.poi_size and self.d_model:
            poi_emb = self.emb_poi(loc)
            pos_emb = self.emb_pos(pos)
            x_emb = torch.cat((loc_emb, poi_emb, tim_emb, usr_emb, pos_emb), -1)
        elif self.d_model and not self.poi_size:
            pos_emb = self.emb_pos(pos)
            x_emb = torch.cat((loc_emb, tim_emb, usr_emb, pos_emb), -1)
        elif self.poi_size and not self.d_model:
            poi_emb = self.emb_poi(loc)
            x_emb = torch.cat((loc_emb, tim_emb, usr_emb, poi_emb), -1)
        else:
            x_emb = torch.cat((loc_emb, tim_emb, usr_emb), -1)

        x_emb = self.dropout(x_emb)

        # Encoder RNN
        batchsize = loc.shape[0]
        h1 = torch.zeros(1, batchsize, self.encoder_rnn_hidden_size).to(self.device)
        c1 = torch.zeros(1, batchsize, self.encoder_rnn_hidden_size).to(self.device)
        hidden, _ = self.encoder_rnn(x_emb, (h1, c1))

        # param for Gauss distribution
        mean = self.mean_l2(F.relu(self.mean_l1(hidden)))
        std = self.std_l2(F.relu(self.std_l1(hidden)))

        return mean, std 


class DECODER(nn.Module):

    def __init__(self, param, emb):
        super(DECODER, self).__init__()

        # Embedding
        self.emb_usr = emb[2]
        self.d_model = param.d_model if not param.feedback_ban else 0
        self.emb_pos = emb[3] if self.d_model else None

        # Decoder RNN
        self.decoder_rnn_input_size = param.latent_size + param.usr_emb_size + self.d_model
        self.decoder_rnn_hidden_size = param.decoder_rnn_hidden_size
        self.decoder_rnn = nn.LSTM(self.decoder_rnn_input_size, self.decoder_rnn_hidden_size, 1, batch_first=True)

        # Dropout
        self.dropout = nn.Dropout(p=param.dropout)

        # Location decoder
        self.loc_hidden_size1 = param.loc_hidden_size1
        self.loc_hidden_size2 = param.loc_hidden_size2
        self.loc_size = param.loc_size
        self.poi_size = param.poi_size
        self.poi_ban = param.poi_ban
        self.loc_l1 = nn.Linear(self.decoder_rnn_hidden_size, self.loc_hidden_size1)
        self.loc_l2 = nn.Linear(self.loc_hidden_size1, self.loc_hidden_size2)
        self.loc_l3 = nn.Linear(self.loc_hidden_size2, self.loc_size)
        if self.poi_size and not self.poi_ban:
            self.loc_l4 = nn.Linear(self.loc_hidden_size2, self.poi_size)
            self.loc_l5 = nn.Linear(self.poi_size, 1)
            self.POI = param.POI
            self.poi_weight = param.poi_weight

        # Waiting time decoder
        self.tim_hidden_size1 = param.tim_hidden_size1
        self.tim_hidden_size2 = param.tim_hidden_size2
        self.tim_l1 = nn.Linear(self.decoder_rnn_hidden_size, self.tim_hidden_size1)
        self.tim_l2 = nn.Linear(self.tim_hidden_size1, self.tim_hidden_size2)
        self.tim_l3 = nn.Linear(self.tim_hidden_size2, 1)

        self.device = param.device

    def forward(self, usr, pos, z):#, loc_weight, usr_weight):

        # Embedding
        usr_emb = self.emb_usr(usr)
        if self.d_model:
            pos_emb = self.emb_pos(pos)
            z = torch.cat((z, usr_emb, pos_emb), -1)
        else:
            z = torch.cat((z, usr_emb), -1)
        z = self.dropout(z)

        # Decoder RNN
        batchsize = usr.shape[0]
        h1 = torch.zeros(1, batchsize, self.decoder_rnn_hidden_size).to(self.device)
        c1 = torch.zeros(1, batchsize, self.decoder_rnn_hidden_size).to(self.device)
        hidden, _ = self.decoder_rnn(z, (h1, c1))

        # Location decoder 
        lout2 = F.selu(self.loc_l2(F.selu(self.loc_l1(hidden))))
        lout_1 = F.softmax(self.loc_l3(lout2), dim=2)
        if self.poi_size and not self.poi_ban:
            lout_2 = F.softmax(self.loc_l5(F.selu(self.loc_l4(lout2)).unsqueeze(-2) * torch.tensor(self.POI).to(self.device)).squeeze(-1), dim=1)
            lout = (1 - self.poi_weight) * lout_1 + self.poi_weight * lout_2
        else:
            lout = lout_1

        # Waiting time decoder
        tout = torch.exp(self.tim_l3(F.relu(self.tim_l2(F.relu(self.tim_l1(hidden)))))).squeeze(-1)

        return lout, tout


class VAE(nn.Module):

    def __init__(self, param):

        super().__init__()
        # Embedding
        self.emb_loc = LOC_EMB(param)
        self.emb_tim = TIM_DIFF_EMB(param)
        self.emb_usr = USR_EMB(param)
        self.emb_pos = ABS_TIM_EMB(param)
        if param.poi_size:
            self.emb_poi = POI_EMB(param)
            emb = [self.emb_loc, self.emb_tim, self.emb_usr, self.emb_pos, self.emb_poi]
        else:
            emb = [self.emb_loc, self.emb_tim, self.emb_usr, self.emb_pos]
        
        self.encoder = ENCODER(param, emb)
        self.decoder = DECODER(param, emb)
        self.dropout = nn.Dropout(p=param.dropout)

        self.latent_size = param.latent_size
        self.loc_size = param.loc_size

        self.infer_maxlast = param.infer_maxlast
        self.infer_maxinternal = param.infer_maxinternal
        self.infer_divide = param.infer_divide

        self.learning_rate = param.learning_rate
        self.L2 = param.L2
        self.step_size = param.step_size
        self.gamma = param.gamma
        self.epoches = param.epoches
        self.batchsize = param.batchsize

        self.step = 0
        self.infer_maxlength = 1000

        self.save_path = param.save_path
        self.device = param.device
        self.feedback_ban = param.feedback_ban
        # Parameters initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight)

    def forward(self, inseq):

        # Encoder
        mean, std = self.encoder(inseq['usr'], inseq['loc'], inseq['tim'], inseq['pos'])

        # Sampling
        z = torch.randn_like(std).to(self.device)
        z = z * std + mean
        z = self.dropout(z)

        # Decoder
        lout, tout = self.decoder(inseq['usr'], inseq['pos'], z)

        return mean, std, lout, tout

    @staticmethod
    def loss(step, mean, std, lout, tout, inseq):

        KL = torch.mean(0.5 * torch.sum((std ** 2) - 1 - torch.log(std ** 2) + (mean ** 2), dim=-1))
        criterion = nn.NLLLoss()
        LL_L = criterion(torch.log(lout.swapaxes(1,2)), inseq['loc'])
        LL_T = -torch.mean(torch.log(1 - torch.exp(-tout / 1000)) - (tout / 1000) * inseq['tim'])

        # Train the decoder first
        weight = 1
        # weight = torch.tensor(1 / (1 + np.exp(-1 * (step - 10)))).to(VAE.device)
        LOSS = LL_T + weight * (LL_L + KL)

        return KL, LL_L, LL_T, LOSS

    def sample(self, lout, tout, last_loc=-1):
        Lambda = tout[0][-1].squeeze().cpu().detach().numpy()
        t = np.random.exponential(1000 / Lambda, size=1)[0]
        if t < 1:
            t = 1
        # while t < 1 or t > self.infer_maxinternal:
            # t = np.random.exponential(1000 / Lambda, size=1)[0]
        prob = lout[0][-1].cpu().detach().numpy()
        if last_loc >= 0:
            prob[last_loc] = 0
        l = np.random.choice(list(range(self.loc_size)), p=prob/sum(prob))
        return l, t

    def inference_withouot_feedback(self, user):

        # Sampling
        z = torch.randn(self.infer_maxlength, self.latent_size).to(self.device)
        usr = user * torch.ones(self.infer_maxlength, dtype=torch.long).to(self.device)

        # Decoder & Generate
        lout1, tout1 = self.decoder(usr[0].view(1,1), torch.zeros((1,1), dtype=torch.double).to(self.device), z[0].view(1,1,-1))
        l, t = self.sample(lout1, tout1)
        X = {'loc': [], 'tim': [t], 'sta': []}
        last_location = int(l)
        lout, tout = self.decoder(usr.unsqueeze(0), usr.unsqueeze(0), z.unsqueeze(0))
        
        for i in range(1, self.infer_maxlength):
            l, t = self.sample(lout[0][:i].unsqueeze(0), tout[0][:i].unsqueeze(0), last_location)
            if X['tim'][-1] + t >= self.infer_maxlast:
                break  
            X['tim'].append(X['tim'][-1] + t)
            X['loc'].append(l)
            X['sta'].append(t)
            last_location = int(X['loc'][-1])

        X['tim'] = np.array(X['tim'])[:-1]
        X['sta'] = np.array(X['sta'])
        X['loc'] = np.array(X['loc'])
        return X

    def inference(self, user):

        # Sampling
        z = torch.randn(self.infer_maxlength, self.latent_size).to(self.device)
        usr = user * torch.ones(self.infer_maxlength, dtype=torch.long).to(self.device)

        # Decoder & Generate
        lout1, tout1 = self.decoder(usr[0].view(1,1), torch.zeros((1,1), dtype=torch.double).to(self.device), z[0].view(1,1,-1))
        l, t = self.sample(lout1, tout1)
        X = {'loc': [], 'tim': [t], 'sta': []}
        last_location = int(l)

        for i in range(1, self.infer_maxlength):
            time = torch.tensor([X['tim']], dtype=torch.double).to(self.device)
            louti, touti = self.decoder(usr[:i].unsqueeze(0), time, z[:i].unsqueeze(0))
            l, t = self.sample(louti, touti, last_location)
            if X['tim'][-1] + t >= self.infer_maxlast:
                break  
            X['tim'].append(X['tim'][-1] + t)
            X['loc'].append(l)
            X['sta'].append(t)
            last_location = int(X['loc'][-1])

        X['tim'] = np.array(X['tim'])[:-1]
        X['sta'] = np.array(X['sta'])
        X['loc'] = np.array(X['loc'])
        return X

    def load(self, cp):
        self.load_state_dict(torch.load(cp))
        print("Load model from %s" % cp )

    def save(self, cp):
        torch.save(self.state_dict(), cp)
        print("Model saved as %s" % cp)

    def test_data_prepare(self, data, load_checkpoint = None):

        if load_checkpoint != None:
            self.load(load_checkpoint)

        output_sequence = {}
        for user in data.REFORM['test']:
            
            print('Inferencing for User' + str(user))
            output_sequence[user] = {}

            # Get the undivided sequence
            undivided = self.inference(user) if not self.feedback_ban else self.inference_withouot_feedback(user)

            # Divide it by day/ year
            divide = [0]
            for key in range(len(undivided['tim']) - 1):
                A = self.infer_divide
                if (undivided['tim'][key] // A) != (undivided['tim'][key + 1] // A):
                    divide.append(key + 1)

            for traj, key in enumerate(divide):
                Key = divide[traj + 1] if (traj + 1) < len(divide) else len(undivided['tim'])
                output_sequence[user][traj] = {'loc': undivided['loc'][key: Key], 'tim': undivided['tim'][key: Key], 'sta': undivided['sta'][key: Key]}

        data.GENDATA.append(output_sequence)

    def train(self, trainset, validset, testset = None):
        print('Start Training')
        loss_record = {}
        valid_record = {}
        self.step = 0
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,  weight_decay=self.L2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

        for epoch in range(self.epoches + 1):
            print('Epoch: %s' % (epoch))

            optimizer.zero_grad()
            loss_record[epoch] = {"LOSS": [], "KL": [], "LL_T": [], "LL_L": []}

            mydataloader = DataLoader(trainset, batch_size=self.batchsize, shuffle=True, num_workers=0, collate_fn=mycollatefunc)
            for idx, bat in enumerate(mydataloader):


                # Input
                loc = bat['loc'].clone().detach().long().to(self.device)
                tim = bat['sta'].clone().detach().long().to(self.device)
                usr = bat['usr'].clone().detach().long().to(self.device)
                pos = bat['tim'].clone().detach().double().to(self.device)

                inseq = {'usr': usr, 'loc': loc, 'tim': tim, 'pos': pos}

                # Output
                mean, std, lout, tout = self.forward(inseq)

                # Loss
                KL, LL_L, LL_T, LOSS = VAE.loss(self.step, mean, std, lout, tout, inseq)

                # Backward
                LOSS.backward()
                optimizer.step()

                # Loss record
                loss_record[epoch]['KL'].append(KL.item())
                loss_record[epoch]['LL_L'].append(LL_L.item())
                loss_record[epoch]['LL_T'].append(LL_T.item())
                loss_record[epoch]['LOSS'].append(LOSS.item())

                print('Index: %s, KL: %s, LL_L: %s, LL_T: %s, ELBO: %s' % (idx, loss_record[epoch]['KL'][-1],
                                                                loss_record[epoch]['LL_L'][-1],
                                                                loss_record[epoch]['LL_T'][-1],
                                                                loss_record[epoch]['LOSS'][-1]))


            
            self.step += 1

            if epoch % 1 == 0:

                print('Start Validation')

                valid_record[epoch] = {"LOSS": [], "KL": [], "LL_T": [], "LL_L": []}

                mydataloader = DataLoader(validset, batch_size=self.batchsize, shuffle=True, num_workers=0, collate_fn=mycollatefunc)
                for bat in mydataloader:

                    # Input
                    loc = bat['loc'].clone().detach().long().to(self.device)
                    tim = bat['sta'].clone().detach().long().to(self.device)
                    usr = bat['usr'].clone().detach().long().to(self.device)
                    pos = bat['tim'].clone().detach().double().to(self.device)

                    inseq = {'usr': usr, 'loc': loc, 'tim': tim, 'pos': pos}

                    # Output
                    mean, std, lout, tout = self.forward(inseq)

                    # Loss
                    KL, LL_L, LL_T, LOSS = VAE.loss(self.step, mean, std, lout, tout, inseq)

                    # Loss record
                    valid_record[epoch]['KL'].append(KL.item())
                    valid_record[epoch]['LL_L'].append(LL_L.item())
                    valid_record[epoch]['LL_T'].append(LL_T.item())
                    valid_record[epoch]['LOSS'].append(LOSS.item())

                print('Epoch: %s, KL: %s, LL_L: %s, LL_T: %s, ELBO: %s' % (epoch, np.mean(valid_record[epoch]['KL']),
                                                                    np.mean(valid_record[epoch]['LL_L']),
                                                                    np.mean(valid_record[epoch]['LL_T']),
                                                                    np.mean(valid_record[epoch]['LOSS'])))

            # Reduce the learning rate
            scheduler.step()

            '''
            # Evaluate the performance and write it down
            if epoch % 20 == 0 and testset != None:
                self.test_data_prepare(testset.dataset)
            '''

        self.save(self.save_path + 'data/Model')
        return loss_record, valid_record

    def test(self, testset):

        print('Start Testing')
        test_record = {"LOSS": [], "KL": [], "LL_T": [], "LL_L": []}
        
        mydataloader = DataLoader(testset, batch_size=self.batchsize, shuffle=True, num_workers=0, collate_fn=mycollatefunc)
        for bat in mydataloader:

            # Input
            loc = bat['loc'].clone().detach().long().to(self.device)
            tim = bat['sta'].clone().detach().long().to(self.device)
            usr = bat['usr'].clone().detach().long().to(self.device)
            pos = bat['tim'].clone().detach().double().to(self.device)

            inseq = {'usr': usr, 'loc': loc, 'tim': tim, 'pos': pos}

            # Output
            mean, std, lout, tout = self.forward(inseq)

            # Loss
            KL, LL_L, LL_T, LOSS = VAE.loss(self.step, mean, std, lout, tout, inseq)

            # Loss record
            test_record['KL'].append(KL.item())
            test_record['LL_L'].append(LL_L.item())
            test_record['LL_T'].append(LL_T.item())
            test_record['LOSS'].append(LOSS.item())

        print('KL: %s, LL_L: %s, LL_T: %s, ELBO: %s' % (np.mean(test_record['KL']),
                                                        np.mean(test_record['LL_L']),
                                                        np.mean(test_record['LL_T']),
                                                        np.mean(test_record['LOSS'])))
        
        self.test_data_prepare(testset.dataset)

        return test_record

    def loss_plot(self, loss_record, valid_record, test_record):

        KL = [np.mean(loss_record[epoch]['KL']) for epoch in loss_record]
        LL_L = [np.mean(loss_record[epoch]['LL_L']) for epoch in loss_record]
        LL_T = [np.mean(loss_record[epoch]['LL_T']) for epoch in loss_record]
        LOSS = [np.mean(loss_record[epoch]['LOSS']) for epoch in loss_record]

        valid_KL = [np.mean(valid_record[epoch]['KL']) for epoch in valid_record]
        valid_LL_L = [np.mean(valid_record[epoch]['LL_L']) for epoch in valid_record]
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

    def run(self, trainset, validset, testset):
        loss_record, valid_record = self.train(trainset, validset, testset)
        test_record = self.test(testset)
        self.loss_plot(loss_record, valid_record, test_record)        
