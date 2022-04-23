# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
Created on Sep 15th 18:05:53 2020

Baseline 1: Semi-Markov Model(SMM)
"""
import numpy as np

def SEMIMARKOV(data, MAX_LENGTH = 1000):

    Markov = {}
    Poisson = {}
    TimeDelta = {}
    for user in data:

        # Recode the location for each user to avoid the influence of those places not visited by them
        locations = np.array([])
        for traj in data[user]:
            locations = np.append(locations, data[user][traj]['loc'])
        locations = np.sort(np.unique(locations))
        trans = lambda x:np.where(locations == x)[0][0]
        detrans = lambda x:locations[x]

        max_locs = locations.size
        real_data = np.array([])
        time_interval = np.array([])
        for traj in data[user]:
            real_data = np.append(real_data, np.array([trans(i) for i in data[user][traj]['loc']]))
            time_interval = np.append(time_interval, data[user][traj]['sta'])
        real_data = real_data.astype(int)

        # Calculate the parameterss
        transfer = np.zeros((max_locs, max_locs))
        time_count = np.zeros((max_locs, max_locs))
        Lambda = np.zeros((max_locs, max_locs))

        LAMBDA = (len(time_interval) - 1) / sum(time_interval[1:])
        alpha = 1
        beta = alpha / LAMBDA

        for i in range(real_data.shape[0] - 1):
            r = real_data[i]
            c = real_data[i + 1]
            transfer[r, c] += 1
            time_count[r, c] += time_interval[i + 1]

        for i in range(max_locs):
            Lambda[i, :] = (transfer[i, :] + alpha) / (time_count[i, :] + beta)
            transfer[i, :] = (transfer[i, :] + 0.1)

        # Sample the first point
        Prob = np.sum(transfer, axis=0)
        Prob = Prob / sum(Prob)
        s = np.random.choice(max_locs, 1, p=Prob)
        s = s[0]
        a = np.random.exponential(1 / LAMBDA, size=1).item()
        Markov[user] = [detrans(s)]
        Poisson[user] = [a]
        TimeDelta[user] = []

        # Generate the whole trajectory given the first point
        for j in range(MAX_LENGTH):
            P = transfer[s, :] / sum(transfer[s, :])
            s = np.random.choice(max_locs, 1, p=P)
            s = s[0]
            a = np.random.exponential(1 / Lambda[trans(Markov[user][-1]), s], size=1)
            if (Poisson[user][j] + a.item()) >= 30*24*60:
                TimeDelta[user].append(a.item())
                break
            Markov[user].append(detrans(s))
            Poisson[user].append(Poisson[user][j] + a.item())
            TimeDelta[user].append(a.item())
            if j == MAX_LENGTH - 2:
                P = transfer[s, :] / sum(transfer[s, :])
                s = np.random.choice(max_locs, 1, p=P)
                s = s[0]
                a = np.random.exponential(1 / Lambda[trans(Markov[user][-1]), s], size=1)
                TimeDelta[user].append(a.item())


        Markov[user] = np.array(Markov[user])
        Poisson[user] = np.array(Poisson[user])
        TimeDelta[user] = np.array(TimeDelta[user])

    # Divide the preliminarily processed sequence into several trajectories that fit our model.
    Semi_Markov = {}
    for user in Markov:
        Semi_Markov[user] = {}

        divide = [0]
        for key in range(len(Poisson[user]) - 1):
            A = 7 * 24 * 60
            if (Poisson[user][key] // A) != (Poisson[user][key + 1] // A):
                divide.append(key + 1)

        for traj, key in enumerate(divide):
            Key = divide[traj+1] if (traj+1) < len(divide) else len(Poisson[user])
            Semi_Markov[user][traj] = {'loc': Markov[user][key: Key], 'tim': Poisson[user][key: Key], 'sta': TimeDelta[user][key: Key]}

    return Semi_Markov
