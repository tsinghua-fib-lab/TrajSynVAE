# TrajSynVAE


## Baselines

* **TimeGeo**: The method built based on the explore and preferential return (EPR) model [PNAS 2016].
  * Directory: code/baselines.py
* **Semi-Markov**: In this model, the waiting time is modeled by the exponential distribution. Dirichlet prior and gamma prior are used to model the transition matrix and the intensity of the waiting time to implement a Bayesian inference [TVT 2016].
  * Directory: code/baselines.py
* **LSTM**: This model directly predicts the next locations and waiting time based on the LSTM network, and the prediction results are utilized as the synthesized trajectories [CVPR 2018].
  * Directory: code/lstm.py
* **Hawkes**: This model is a widely used classical temporal point process, where an occurred data point will influence the intensity function of future points [QF 2016].
  * Directory: code/baselines.py
* **MoveSim**: The model proposed to synthesize human trajectories based on GAN, which introduces prior knowledge and physical regularities to the SeqGAN model [KDD 2020]
  * Link: https://github.com/FIBLAB/MoveSim


REFERENCES
==========

[KDD 2020] J. Feng, Z. Yang, F. Xu, H. Yu, M. Wang, Y. Li, "Learning to simulate human mobility", in Proc. KDD, 2020.

[PNAS 2016] S. Jiang, Y. Yang, S. Gupta, D. Veneziano, S. Athavale, and M. C. González , "The TimeGeo modeling framework for urban mobility without travel surveys", PNAS, 2016.

[TVT 2016] L. A. Maglaras and D. Katsaros, "Social clustering of vehicles based on semi-markov processes," IEEE Transactions on Vehicular Technology (TVT), 2016.

[QF 2016] E. Bacry, T. Jaisson, and J. Muzy, "Estimation of slowly decreasing hawkes kernels: application to high-frequency order book dynamics," Quantitative Finance, 2016.

[CVPR 2018] Y. Abu Farha, A. Richard, and J. Gall, "When will you do what?-anticipating temporal occurrences of activities,” in Proc. CVPR, 2018.
