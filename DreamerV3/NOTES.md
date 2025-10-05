v1: This shit is made to be SIMPLE. Many cuts were made based off of what is an "improvement" and what is part of Dreamer's core World Model functionality.

Differences
 - Training minibatch should be mix of online queue and replay buffer; currently uses only replay buffer
 - Recurrent representation is not nearly as deep/wide as it should be (i think)
 - Optimizer used is Adam with simple gradnorm clipping; in paper is LaProp w/ adaptive grad clipping
 - L2/MSE loss instead of symlog
 - Reward net as gaussian instead of bins
 - Drop-in SAC with no modifications as actor-critic instead of DreamerV3 custom
 - random critic init, not zeros
 - Original paper uses v(s) instead of q(s). For return calculation RT, instead of using vT as the estimated return, using a single sample a ~ pi(a|s) for q(s, a) instead as spinning up SAC does
 - Smoothed (EMA) target Q nets instead of EMA'd returns
 - In original paper, critic target Q value is a mix of critic value v(s) and trajectory return R with balance factor lambda. this not implemented (yet)
 - Actor target in original paper uses advantage, so this one does too, except using E[Q] + H instead of E[V]

Concerns (not necessarily problems. but they could be)
 - When a latent state h is stored in replay buffer for t in [1, T], h for any t > 1 can be updated and stored into buffer, but h1 cannot. The issue is that h1 will remain stagnant until the experience is overriden. 

Notes
 - When training WM, training is done like a rollout, and full BPTT is used from hT to h1
 - In WM training rollout, z grads are stopped during BPTT
 - Grads betwen agent and WM do NOT interact
 - Actions taken during actor-critic trajectory generation do not have grad connection to training. Training is treated as independent of trajectory gen.
 - WM and agent do NOT train on the same batch. I don't see any reason they need to.
 - Covariance matrix of policy assumed to be diagonal (e.g. no covariance terms)
 - The fast-updating q-network is never actually used. Only the target q is used. This is probably why the paper opts for EMA'd returns. But, because the target q's are already implemented, I'm keeping them for ease. It should help with maximization bias a little too.
