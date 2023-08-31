"""
Created on Sep 15th 18:05:53 2020

Author: Qizhong Zhang

Data-based baseline: LSTM model
"""

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
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
Embedding Modules: Sojourn time, location id, user id
'''

class TIM_DIFF_EMB(nn.Module):

    def __init__(self, param):
        super().__init__()
        self.tim_emb_type = param.tim_emb_type
        self.tim_size = param.tim_size
        self.tim_emb_size = param.tim_emb_size
        self.emb_tim = nn.Linear(1, self.tim_emb_size, bias=False)

    def forward(self, x):
        return self.emb_tim(torch.log(x.unsqueeze(-1) + 1e-10))


class LOC_EMB(nn.Module):

    def __init__(self, param):
        super().__init__()

        self.loc_size = param.loc_size + 1
        self.loc_emb_size = param.loc_emb_size
        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size, padding_idx=0)
        
    def forward(self, x):
        return self.emb_loc(x)


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

'''
Basic LSTM model
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


class lstm(nn.Module):

    def __init__(self, param):
        super(lstm, self).__init__()

        # Embedding
        self.emb_loc = LOC_EMB(param)
        self.emb_usr = USR_EMB(param)
        self.emb_tim = TIM_DIFF_EMB(param)

        # Decoder RNN
        self.decoder_rnn_input_size = param.loc_emb_size + param.usr_emb_size + param.tim_emb_size
        self.decoder_rnn_hidden_size = param.decoder_rnn_hidden_size
        self.rnn_type = param.rnn_type
        if self.rnn_type == 'LSTM':
            self.decoder_rnn = nn.LSTM(self.decoder_rnn_input_size, self.decoder_rnn_hidden_size, 1, batch_first=True)
        else:
            self.decoder_rnn = nn.GRU(self.decoder_rnn_input_size, self.decoder_rnn_hidden_size, 1, batch_first=True)

        # Dropout
        self.dropout = nn.Dropout(param.dropout)

        # Location decoder
        self.loc_hidden_size1 = param.loc_hidden_size1
        self.loc_hidden_size2 = param.loc_hidden_size2
        self.loc_size = param.loc_size + 1
        self.loc_l1 = nn.Linear(self.decoder_rnn_hidden_size, self.loc_hidden_size1)
        self.loc_l2 = nn.Linear(self.loc_hidden_size1, self.loc_hidden_size2)
        self.loc_l3 = nn.Linear(self.loc_hidden_size2, self.loc_size)

        # Sojourn time decoder
        self.tim_hidden_size1 = param.tim_hidden_size1
        self.tim_hidden_size2 = param.tim_hidden_size2
        self.tim_l1 = nn.Linear(self.decoder_rnn_hidden_size, self.tim_hidden_size1)
        self.tim_l2 = nn.Linear(self.tim_hidden_size1, self.tim_hidden_size2)
        self.tim_l3 = nn.Linear(self.tim_hidden_size2, 1)

        self.device = param.device

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                initialize_linear_layer(m)
            if isinstance(m, nn.LSTM):
                initialize_rnn(m)
            if isinstance(m, nn.GRU):
                initialize_rnn(m, 'GRU')

    def forward(self, usr, loc, sta, lengths = None):

        # Embedding
        usr_emb = self.emb_usr(usr)
        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(sta)
        z = torch.cat((tim_emb, usr_emb, loc_emb), -1)

        # RNN
        lengths = lengths if lengths is not None else [z.shape[1]] * z.shape[0]
        packed_input = pack_padded_sequence(input=z, lengths=lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.decoder_rnn(packed_input)
        hidden, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=z.shape[1])

        # Dropout
        hidden = self.dropout(hidden)

        # Location decoder 
        lout = F.log_softmax(self.loc_l3(F.leaky_relu(self.loc_l2(F.leaky_relu(self.loc_l1(hidden))))), dim=2)

        # Waiting time decoder
        tout = self.tim_l3(F.leaky_relu(self.tim_l2(F.leaky_relu(self.tim_l1(hidden))))).squeeze(-1)

        return lout, tout - np.log(100)

'''
Inference and Generating
'''

class LSTMMODEL(nn.Module):

    def __init__(self, param):

        super().__init__()
        self.model = lstm(param)

        # Inference and generation hyperparameters
        self.loc_size = param.loc_size + 1
        self.USERLIST = param.USERLIST
        self.user_indicator = np.ones((param.USERLIST.shape[0]+1, param.loc_size+1))
        self.ntrajs = param.ntrajs
        self.infer_maxlast = param.infer_maxlast
        self.save_path = param.save_path

        # Training hyperparameter
        self.learning_rate = param.learning_rate
        self.L2 = param.L2
        self.step_size = param.step_size
        self.gamma = param.gamma
        self.epoches = param.epoches
        self.batchsize = param.batchsize

        self.device = param.device

    def forward(self, inseq):

        lout, tout = self.model(inseq['usr']+1, inseq['loc']+1, inseq['tim'], inseq['lengths'])
        lout = self.locprob_filter(lout, inseq['usr'])
        return lout, tout

    # Loss function: negative log-likelihoods
    def loss(self, lout, tout, loc, tim, lengths):

        mask = pad_sequence([torch.tensor([1] * int(l)) for l in lengths], batch_first=True).to(self.device)
        percentage = torch.sum(mask) / (mask.shape[0] * mask.shape[1])
        LL_L = torch.mean(nn.NLLLoss(reduction='none')(lout.swapaxes(1,2), loc + 1) * mask) / percentage
        LL_T = -torch.mean((tout - torch.exp(tout) * tim) * mask) / percentage
        LOSS = LL_T + LL_L
        return LL_L, LL_T, LOSS

    # Only the locations in the training set of one user will be involved in corresponding inference and generation for this user
    def locprob_filter(self, lout, usr, training = True):

        user_encoding = np.array([[np.where(np.append(-1, self.USERLIST) == u.item())[0][0] for u in user] for user in usr]).astype(int)
        weights = torch.tensor(np.array([self.user_indicator[np.ix_(usrs)] for usrs in user_encoding])).to(self.device)
        l0 = lout - torch.log(weights) if training else lout.masked_fill(weights != 1, float('-inf'))
        l1 = l0 - torch.logsumexp(l0, dim=2, keepdim=True)
        return l1
   
    def load(self, cp):
        self.load_state_dict(torch.load(cp))
        print("Load model from %s" % cp )

    def save(self, cp):
        torch.save(self.state_dict(), cp)
        print("Model saved as %s" % cp)

    # Sample the sojourn time and location time 
    def sample(self, lout, tout, last_loc, ntrajs = 7): 

        Lambda = torch.exp(tout[:, -1]).unsqueeze(1).cpu().detach().numpy()
        def truncated_exponential_samples(size, lower, upper, rate):
            uniform_samples = np.random.rand(*size)
            truncated_samples = -np.log(1 - uniform_samples * (1 - np.exp(-rate * (upper - lower)))) / rate + lower
            return truncated_samples
        t = np.squeeze(truncated_exponential_samples((ntrajs, 1), 10, self.infer_maxlast, Lambda))

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
            # Get the starting point
            usr = (user + 1) * torch.ones((ntrajs, 144), dtype=torch.long).to(self.device)
            last_location = np.zeros(ntrajs).astype(int)

            start_loc = torch.tensor(last_location).unsqueeze(1).to(self.device).long()
            start_pos = torch.ones((ntrajs,1), dtype=torch.double).to(self.device) * 10
            lout1, tout1 = self.model(usr[:,0].unsqueeze(1), start_loc, start_pos)
            lout1 = self.locprob_filter(lout1, usr[:,0].unsqueeze(1)-1)

            l, t = self.sample(lout1, tout1, last_location, ntrajs = ntrajs)
            X = {'loc': [l], 'tim': [], 'sta': [t]}
            last_location = l.astype(int)

            # Generating
            for i in range(1, 145):
                time = torch.tensor(np.array(X['sta']).T).double().to(self.device)
                location = torch.tensor(np.array(X['loc']).T).long().to(self.device)
                louti, touti = self.model(usr[:, :i], location, time)
                louti = self.locprob_filter(louti, usr[:, :i]-1)

                l, t = self.sample(louti, touti, last_location)
                X['loc'].append(l)
                X['sta'].append(t)

                last_location = np.array(X['loc'])[-1].astype(int)
                if np.min(np.array(X['sta']).sum(axis=0)) >= self.infer_maxlast:
                    break  

            # Get the output sequence
            X['tim'] = np.cumsum(X['sta'], 0)
            output = {}
            for i in range(ntrajs):
                id = np.where(np.array(X['tim'])[:, i] >= self.infer_maxlast)[0][0]
                output[i] = {'loc': np.array(X['loc'])[1:(1+id), i] - 1,
                             'tim': np.array(X['tim'])[:id, i], 
                             'sta': np.array(X['sta'])[1:(1+id), i]}
                
            return output

    # Generating for all users
    @torch.no_grad()
    def test_data_prepare(self, data):
        
        self.eval()
        with torch.no_grad():
            output_sequence = {}
            gen_bar = tqdm(data.REFORM['test'])
            for user in gen_bar:
                gen_bar.set_description("Generating trajectories for user: {}".format(user))
                output_sequence[user] = self.inference(user, self.ntrajs)                
            data.GENDATA.append(output_sequence)
            np.save(self.save_path + 'data/original.npy', output_sequence)

    # Count for all locations that appear in the historical trajectories for each user
    def location_constraints(self, data):
        user_indicator = np.ones((self.USERLIST.shape[0]+1, self.loc_size))
        weights = np.ones((self.USERLIST.shape[0]+1, self.loc_size))
        for userid, data_user in data.items():
            count = np.bincount(np.concatenate([traj['loc']+1 for traj in data_user.values()]))
            count = np.append(count, np.zeros(self.loc_size - count.shape[0]))
            weights[np.where(self.USERLIST == userid)[0][0]+1] = count
            user_indicator[(np.where(self.USERLIST == userid)[0][0]+1, count == 0)] = 1e6
        weights[weights > 0] = np.log(weights[weights > 0]) + 1
        return user_indicator, weights

    # Training
    def Train(self, trainset, validset):

        self.train()

        # Set optimizer and scheduler
        print('Start Training')
        self.user_indicator, self.loc_weights = self.location_constraints(trainset.dataset.REFORM['train'])
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,  weight_decay=self.L2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

        loss_record, valid_record = {}, {}
        for epoch in range(1, self.epoches + 1):
            optimizer.zero_grad()
            loss_record[epoch] = {"LOSS": [], "LL_T": [], "LL_L": []}

            mydataloader = DataLoader(trainset, batch_size=self.batchsize, shuffle=True, num_workers=0, collate_fn=mycollatefunc)
            train_bar = tqdm(enumerate(mydataloader))
            for idx, bat in train_bar:

                # Input
                loc = bat['loc'].clone().detach().long().to(self.device)
                tim = bat['sta'].clone().detach().double().to(self.device)
                usr = bat['usr'].clone().detach().long().to(self.device)
                lengths = np.array(bat['lengths'])
                lengths = np.where(lengths >= lengths.max(), lengths.max() - 1, lengths)
                inseq = {'usr': usr[:, :-1], 'loc': loc[:, :-1], 'tim': tim[:, :-1], 'lengths': lengths}

                # Output
                lout, tout = self.forward(inseq)

                # Loss
                LL_L, LL_T, LOSS = self.loss(lout, tout, loc[:, 1:], tim[:, 1:], lengths)

                # Backward
                LOSS.backward()
                optimizer.step()

                # Loss record
                loss_record[epoch]['LL_L'].append(LL_L.item())
                loss_record[epoch]['LL_T'].append(LL_T.item())
                loss_record[epoch]['LOSS'].append(LOSS.item())

                train_bar.set_description('Training epoch: {}, Index: {}, LL_L: {:.5f}, LL_T: {:.5f}, LOSS: {:.5f}'.format\
                                           (epoch, idx, np.mean(loss_record[epoch]['LL_L']),
                                            np.mean(loss_record[epoch]['LL_T']),
                                            np.mean(loss_record[epoch]['LOSS'])))

            if epoch % 1 == 0:

                self.eval()

                valid_record[epoch] = {"LOSS": [], "LL_T": [], "LL_L": []}
                with torch.no_grad():
                    mydataloader = DataLoader(validset, batch_size=self.batchsize, shuffle=True, num_workers=0, collate_fn=mycollatefunc)
                    valid_bar = tqdm(enumerate(mydataloader))
                    for idx, bat in valid_bar:

                        # Input
                        loc = bat['loc'].clone().detach().long().to(self.device)
                        tim = bat['sta'].clone().detach().double().to(self.device)
                        usr = bat['usr'].clone().detach().long().to(self.device)
                        lengths = np.array(bat['lengths'])
                        lengths = np.where(lengths >= lengths.max(), lengths.max() - 1, lengths)
                        inseq = {'usr': usr[:, :-1], 'loc': loc[:, :-1], 'tim': tim[:, :-1], 'lengths': lengths}

                        # Output
                        lout, tout = self.forward(inseq)

                        # Loss
                        LL_L, LL_T, LOSS = self.loss(lout, tout, loc[:, 1:], tim[:, :1], lengths)

                        # Loss record
                        valid_record[epoch]['LL_L'].append(LL_L.item())
                        valid_record[epoch]['LL_T'].append(LL_T.item())
                        valid_record[epoch]['LOSS'].append(LOSS.item())


                        valid_bar.set_description('Validation epoch: {}, Index: {}, LL_L: {:.5f}, LL_T: {:.5f}, LOSS: {:.5f}'.format\
                                            (epoch, idx, np.mean(valid_record[epoch]['LL_L']),
                                                np.mean(valid_record[epoch]['LL_T']),
                                                np.mean(valid_record[epoch]['LOSS'])))

                self.train()

            # Reduce the learning rate
            scheduler.step()

            # checkpoing
            if epoch % 10 == 0:
                self.save(self.save_path + 'data/Model.pth')

        self.save(self.save_path + 'data/Model.pth')
        return loss_record, valid_record

    # Testing
    def test(self, testset):

        self.eval()

        with torch.no_grad():
            test_record = {"LOSS": [], "LL_T": [], "LL_L": []}
            mydataloader = DataLoader(testset, batch_size=self.batchsize, shuffle=True, num_workers=0, collate_fn=mycollatefunc)
            test_bar = tqdm(enumerate(mydataloader))
            for idx, bat in test_bar:

                # Input
                loc = bat['loc'].clone().detach().long().to(self.device)
                tim = bat['sta'].clone().detach().double().to(self.device)
                usr = bat['usr'].clone().detach().long().to(self.device)
                lengths = np.array(bat['lengths'])
                lengths = np.where(lengths >= lengths.max(), lengths.max() - 1, lengths)
                inseq = {'usr': usr[:, :-1], 'loc': loc[:, :-1], 'tim': tim[:, :-1], 'lengths': lengths}

                # Output
                lout, tout = self.forward(inseq)

                # Loss
                LL_L, LL_T, LOSS = self.loss(lout, tout, loc[:, 1:], tim[:, 1:], lengths)

                # Loss record
                test_record['LL_L'].append(LL_L.item())
                test_record['LL_T'].append(LL_T.item())
                test_record['LOSS'].append(LOSS.item())
            
                test_bar.set_description('Testing index: {}, LL_L: {:.5f}, LL_T: {:.5f}, LOSS: {:.5f}'.format\
                                            (idx, np.mean(test_record['LL_L']),
                                                np.mean(test_record['LL_T']),
                                                np.mean(test_record['LOSS'])))
        
        # Generate data for further study
        self.test_data_prepare(testset.dataset)

        return test_record

    # Plot the losses
    def loss_plot(self, loss_record, valid_record, test_record):

        LL_L = [np.mean(loss_record[epoch]['LL_L']) for epoch in loss_record]
        LL_T = [np.mean(loss_record[epoch]['LL_T']) for epoch in loss_record]
        LOSS = [np.mean(loss_record[epoch]['LOSS']) for epoch in loss_record]

        valid_LL_L = [np.mean(valid_record[epoch]['LL_L']) for epoch in valid_record]
        valid_LL_T = [np.mean(valid_record[epoch]['LL_T']) for epoch in valid_record]
        valid_LOSS = [np.mean(valid_record[epoch]['LOSS']) for epoch in valid_record]

        x = np.array([epoch for epoch in loss_record])
        y = np.array([epoch for epoch in valid_record])
    
        plt.figure()

        plt.subplot(131)
        ln1, = plt.plot(x, LL_L, color='red', linewidth=2.0, linestyle='-')
        ln2, = plt.plot(y, valid_LL_L, color='blue', linewidth=2.0, linestyle='-')
        plt.title('LL_L, Test = ' + str(np.around(np.mean(test_record['LL_L']), decimals=3)))
        plt.xlabel('Epoches')
        plt.ylabel('LL_LLoss')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(handles=[ln1, ln2], labels=['Train', 'Valid'])

        plt.subplot(132)
        ln1, = plt.plot(x, LL_T, color='red', linewidth=2.0, linestyle='-')
        ln2, = plt.plot(y, valid_LL_T, color='blue', linewidth=2.0, linestyle='-')
        plt.title('LL_T, Test = ' + str(np.around(np.mean(test_record['LL_T']), decimals=3)))
        plt.xlabel('Epoches')
        plt.ylabel('LL_TLoss')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(handles=[ln1, ln2], labels=['Train', 'Valid'])

        plt.subplot(133)
        ln1, = plt.plot(x, LOSS, color='red', linewidth=2.0, linestyle='-')
        ln2, = plt.plot(y, valid_LOSS, color='blue', linewidth=2.0, linestyle='-')
        plt.title('LOSS, Test = ' + str(np.around(np.mean(test_record['LOSS']), decimals=3)))
        plt.xlabel('Epoches')
        plt.ylabel('Loss')
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
