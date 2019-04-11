import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the dim of feature to be splitted

		self.feature_uniq_split = None # the feature to be splitted


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of
					  corresponding training
					  samples
					  e.g. 		4B4W
					  		  2B2W	2B
					  [[2,2],[4,0]]

			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################
			b = np.array(branches)
			N = np.sum(b)
			bN = b.sum(axis=0)
			pmk = b/bN
			pmk[pmk == 0] = 1 # So it doesn't mess up the log calc (Shouldn't affect anything)
			lpmk = np.log(pmk)
			m = pmk*lpmk
			bentropy = -1*m.sum(axis=0)

			return np.sum((bN/N)*bentropy)

			
		min_entropy = None

		for idx_dim in range(len(self.features[0])):
		############################################################
		# TODO: compare each split using conditional entropy
		#       find the best split
		############################################################
			branch = [f[idx_dim] for f in self.features]
			num_b = len(set(branch))
			#self.labels
			#self.num_cls
			
			# Init to all 0s
			splitted = [[0 for col in range(num_b)] for row in range(self.num_cls)]

			# Have to create ordering for branch value (Cant expect to be in sequence from 0)
			branch_set = list(sorted(set(branch)))

			# For each feature
			for i in range(len(self.labels)):
				splitted[self.labels[i]][branch_set.index(branch[i])] += 1

			entropy = conditional_entropy(splitted)

			# Want to use this split condition
			if min_entropy == None or entropy < min_entropy:
				min_entropy = entropy
				self.dim_split = idx_dim
				self.feature_uniq_split = branch_set


		############################################################
		# TODO: split the node, add child nodes
		############################################################
		
		# Segment into len(self.feature_uniq_split) groups 
		# of features, labels, and num_cls(Stays the same)
		features_group = {}
		labels_group = {}

		# Init
		for i in range(len(self.feature_uniq_split)):
			features_group[i] = []
			labels_group[i] = []

		# Segment
		for i in range(0,len(self.labels)):
			where = self.feature_uniq_split.index(self.features[i][self.dim_split])
			features_group[where].append(self.features[i])
			labels_group[where].append(self.labels[i])


		# Per branch create a child
		for i in range(len(self.feature_uniq_split)):
			child = TreeNode(features_group[i], labels_group[i], self.num_cls)
			
			hashed = {}
			for f in child.features: 
				hashed[str(f)] = "Marked"

			if len(hashed) == 1: # All elements share same features
				child.splittable = False

			self.children.append(child)


		# split the child nodes
		for child in self.children:
			if child.splittable:
				child.split()

		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			# print(feature)
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max



