from __future__ import print_function
import json
import numpy as np
import sys

def forward(pi, A, B, O):
  """
  Forward algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - alpha: A numpy array alpha[j, t] = P(Z_t = s_j, x_1:x_t)
  """
  S = len(pi)
  N = len(O)
  alpha = np.zeros([S, N])
  ###################################################
  # Q3.1 Edit here
  ###################################################
  
  # Pre-processing - Only care about prob of observed observations
  # Form new matrix BPrime[i, k*] w/ k indexing observation at time t
  BPrime = np.take(B, O, axis=1)

  # Have to calculate initial values manually
  alpha[:,0] = np.einsum('i,i->i',pi,BPrime[:,0])

  # Iterate by time
  for t in range(1,N):
    inter = np.einsum('i,ij->ij',alpha[:,t-1],A) # Summation of transition prob * alpha t-1 for every possible prev state
    alpha[:,t] = np.einsum('i,ji->i',BPrime[:,t],inter)  

  return alpha


def backward(pi, A, B, O):
  """
  Backward algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - beta: A numpy array beta[j, t] = P(x_t+1:x_T | Z_t = s_j )
  """
  S = len(pi)
  N = len(O)
  beta = np.zeros([S, N])
  ###################################################
  # Q3.1 Edit here
  ###################################################
  
  # Pre-processing - Only care about prob of observed observations
  # Form new matrix BPrime[i, k*] w/ k indexing observation at time t
  BPrime = np.take(B, O, axis=1)

  # Have to calculate initial values manually
  beta[:,N-1] = np.ones(S)

  # Iterate by time
  for t in range(N-2,-1,-1):
    beta[:,t] = np.einsum('i,i,ji->j',BPrime[:,t+1],beta[:,t+1],A)

  return beta

def seqprob_forward(alpha):
  """
  Total probability of observing the whole sequence using the forward algorithm

  Inputs:
  - alpha: A numpy array alpha[j, t] = P(Z_t = s_j, x_1:x_t)

  Returns:
  - prob: A float number of P(x_1:x_T)
  """
  prob = 0
  ###################################################
  # Q3.2 Edit here
  ###################################################
  
  R,C = alpha.shape

  prob = np.sum(alpha[:,C-1])

  return prob


def seqprob_backward(beta, pi, B, O):
  """
  Total probability of observing the whole sequence using the backward algorithm

  Inputs:
  - beta: A numpy array beta: A numpy array beta[j, t] = P(Z_t = s_j, x_t+1:x_T)
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence
      (in terms of the observation index, not the actual symbol)

  Returns:
  - prob: A float number of P(x_1:x_T)
  """
  prob = 0
  ###################################################
  # Q3.2 Edit here
  ###################################################
  
  # Pre-processing - Only care about prob of observed observations
  # Form new matrix BPrime[i, k*] w/ k indexing observation at time t
  BPrime = np.take(B, O, axis=1)

  prob = np.einsum('i,i,i->',beta[:,0],pi,BPrime[:,0])

  return prob

def viterbi(pi, A, B, O):
  """
  Viterbi algorithm

  Inputs:
  - pi: A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - path: A list of the most likely hidden state path k* (in terms of the state index)
    argmax_k P(s_k1:s_kT | x_1:x_T)
  """
  path = []
  ###################################################
  # Q3.3 Edit here
  ###################################################
  
  # Pre-processing - Only care about prob of observed observations
  # Form new matrix BPrime[i, k*] w/ k indexing observation at time t
  BPrime = np.take(B, O, axis=1)

  S = len(pi)
  N = len(O)
  delta = np.zeros(N)
  path = np.zeros([S, N])

  # Have to manually do first delta
  delta = np.einsum('i,i->i',pi,BPrime[:,0])

  # Traversing thru time
  for t in range(1,N):
    inter1 = np.einsum('i,ij->ij',delta,A)
    inter2 = np.einsum('i,ji->ji',BPrime[:,t],inter1)
    delta = np.max(inter2,axis=0)
    path[:,t-1] = np.argmax(inter2,axis=0)

  # Find most likely end-state
  mIndex = np.argmax(delta)

  path = path[mIndex].astype(int).tolist()

  return path


##### DO NOT MODIFY ANYTHING BELOW THIS ###################
def main():
  model_file = sys.argv[1]
  Osymbols = sys.argv[2]

  #### load data ####
  with open(model_file, 'r') as f:
    data = json.load(f)
  A = np.array(data['A'])
  B = np.array(data['B'])
  pi = np.array(data['pi'])
  #### observation symbols #####
  obs_symbols = data['observations']
  #### state symbols #####
  states_symbols = data['states']

  N = len(Osymbols)
  O = [obs_symbols[j] for j in Osymbols]

  alpha = forward(pi, A, B, O)
  beta = backward(pi, A, B, O)

  prob1 = seqprob_forward(alpha)
  prob2 = seqprob_backward(beta, pi, B, O)
  print('Total log probability of observing the sequence %s is %g, %g.' % (Osymbols, np.log(prob1), np.log(prob2)))

  viterbi_path = viterbi(pi, A, B, O)

  print('Viterbi best path is ')
  for j in viterbi_path:
    print(states_symbols[j], end=' ')

if __name__ == "__main__":
  main()