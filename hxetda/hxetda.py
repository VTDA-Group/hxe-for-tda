import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn import preprocessing
import torch.nn as nn

def masked_softmax(vec, mask, dim=1, epsilon=1e-10):
    masked_vec = vec * mask.float()
    exps = torch.exp(masked_vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    final_vec = masked_exps/masked_sums + ((1-mask) * vec)
    return final_vec

def custom_hier_loss(output, target, target_weights, mask_list, pathlengths):
    final_sum = 0
    alpha = 0.5
    output[:, 0] = 1.0

    for i,mask in enumerate(mask_list):
        output = masked_softmax(output, mask)
    output = output.log()
    output = output * np.exp(-alpha * (pathlengths - 1))
    final_sum = (target_weights * (output*target).sum(dim=1)).mean()
    #final_sum = ((output*target).sum(dim=1)).mean()

    return -final_sum

def calc_class_weights(labels, vertices, all_paths):
	# Get the weights of each class label...
	ulabels, class_counts = np.unique(labels, return_counts = True)
	class_weight_dict = {}
	for ulabel in ulabels:
		count_for_ulabel = 0
		for ulabel2 in ulabels:
			gind = np.where(vertices == ulabel2)
			if ulabel in all_paths[gind[0][0]]:
				count_for_ulabel += class_counts[np.where(ulabels==ulabel2)]

		class_weight_dict[ulabel] = (10/count_for_ulabel)
	return class_weight_dict

#this could take just a graph...
def calc_path_and_mask(G, vertices, root):
	all_paths = np.zeros((len(vertices), len(vertices)))
	new_new_A = np.zeros((len(vertices), len(vertices)))

	# Find root...
	root = 'Object'

	pathlengths = []
	parent_groups = []
	all_paths = []
	for i,node in enumerate(vertices):
		pathlengths.append(len(nx.shortest_path(G, root, node)))
		all_paths.append(nx.shortest_path(G, root, node))
		for thing in nx.shortest_path(G, root, node):
			gind = np.where(thing == np.asarray(vertices, dtype='str'))[0]
			new_new_A[i,gind[0]] = 1
		if i == 0:
			parent_groups.append(-1)
		else:
			parent_groups.append(np.where(np.asarray(vertices, dtype='str')==next(G.predecessors(node)))[0][0])
	#Make parent groups into a set of masks
	mask_list = []
	for pg in np.unique(parent_groups):
		if pg == -1:
			continue
		gind = np.where(parent_groups == pg)
		mask = np.zeros(len(new_new_A))
		mask[gind] = 1
		mask_list.append(torch.tensor(mask, dtype=int))
	y_dict = dict(zip(vertices, new_new_A))
	pathlengths = torch.tensor(pathlengths)

	return all_paths, pathlengths, mask_list, y_dict

def get_prob(input_vec, desired_class, all_paths, vertices, mask_list):
    output = input_vec * 1.0
    output[:,0] = 1.0

    for i,mask in enumerate(mask_list):
        output = masked_softmax(output, mask)
    
    gind = np.where(vertices == desired_class)
    myprob = torch.ones(len(input_vec))
    for thing in all_paths[gind[0][0]]:
        gind2 = np.where(vertices == thing)[0]
        myprob = myprob * output[:,gind2[0]]
    return myprob


class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.output_size  = output_size

            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.Sigmoid()
            self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.fc3 = torch.nn.Linear(self.hidden_size, self.output_size)
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output1 = self.fc2(relu)
            output2 = self.relu(output1)
            output3 = self.fc3(output2)
            return output3