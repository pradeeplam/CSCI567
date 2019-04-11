import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		########################################################
		# TODO: implement "predict"
		########################################################

		zipped = [(self.betas[t]*np.array(self.clfs_picked[t].predict(features))).tolist() for t in range(len(self.clfs_picked))]
		zipped = np.array(zipped).sum(axis=0)
		zipped = zipped.flatten()

		mask = (zipped < 0)

		zipped[mask] = -1.0
		zipped[~mask] = 1.0 

		y = zipped.astype(int)
		y = y.tolist()

		return y

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		############################################################
		# TODO: implement "train"
		############################################################
		N = len(labels)

		y = np.array(labels)

		w_t = np.array([1.0/N]*N)


		for iter in range(self.T):
			h_t = min(self.clfs, key=lambda h:np.sum(w_t*(y != np.array(h.predict(features))).astype(int)))
			
			epsilon_t = np.sum(w_t*(y != np.array(h_t.predict(features))).astype(int))
			
			beta_t = 0.5*np.log((1-epsilon_t)/epsilon_t)
			
			mask = (y == np.array(h_t.predict(features)))

			w_t[mask] *= np.exp(-1.0*beta_t)
			w_t[~mask] *= np.exp(beta_t)

			w_t = w_t/np.sum(w_t)

			self.clfs_picked.append(h_t)
			self.betas.append(beta_t)

					
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)


class LogitBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "LogitBoost"
		return

	def train(self, features: List[List[float]], labels: List[int]):
		############################################################
		# TODO: implement "train"
		############################################################
		N = len(labels)

		y = np.array(labels)

		pi_t = np.array([0.5]*N)
		f_t = np.array([0.0]*N)

		for iter in range(self.T):
			z_t = (0.5*(y+1) - pi_t)/(pi_t*(1-pi_t))

			w_t = pi_t*(1-pi_t)

			h_t = min(self.clfs, key=lambda h:np.sum(w_t*((z_t - np.array(h.predict(features)))**2)))

			f_t += 0.5*np.array(h_t.predict(features))

			pi_t = 1/(1 + np.exp(-2*f_t))

			self.clfs_picked.append(h_t)
			self.betas.append(0.5)
  


	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)
	