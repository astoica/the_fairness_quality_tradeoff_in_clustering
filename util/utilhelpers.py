''' This code is adapted from https://github.com/Seyed2357/Fair-Clustering-Under-Bounded-Cost and used for the repeated FCBC algorithm (Algorithm 2).'''

import numpy as np 

epsilon = 0.0001 

def dot(K, L):
	if len(K) != len(L):
		return 0
	return sum(i[0] * i[1] for i in zip(K, L))




def get_clust_sizes(x_,num_clusters):
	x_ = np.reshape(x_ , (-1,num_clusters)) 
	return np.sum(x_,axis=0)

# find the proportions based on given assignments 
def find_proprtions_two_color(x,num_colors,color_prob,num_clusters):
	x = np.reshape(x , (-1,num_clusters)) 
	proportions = np.zeros(num_clusters)

	for cluster in range(num_clusters):
		proportions[cluster] = dot(x[:,cluster],color_prob)

	div_total = np.sum(x,axis=0)
	div_total[np.where(div_total == 0)]=1 
	proportions_normalized = proportions/div_total

	return proportions_normalized, proportions, np.sum(x,axis=0)



def find_proprtions_two_color_deter(x,num_colors,color_flag,num_clusters):
	x = np.reshape(x , (-1,num_clusters)) 
	proportions = np.zeros(num_clusters)
	proportions_normalized = np.zeros((2,num_clusters))
    
	for cluster in range(num_clusters):
		proportions[cluster] = dot(x[:,cluster],color_flag)

	div_total = np.sum(x,axis=0)
	div_total[np.where(div_total == 0)]=1 
	proportions_normalized[0,:] = proportions/div_total
	proportions_normalized[1,:] = np.ones(num_clusters)- proportions_normalized[0,:] 
	return proportions_normalized, proportions, np.sum(x,axis=0)


# find the proportions based on given assignments, multi-color 
def find_proprtions_multi_color(x,num_colors,prob_vecs,num_clusters):
	x = np.reshape(x , (-1,num_clusters)) 
	div_total = np.sum(x,axis=0)
	div_total[np.where(div_total == 0)]=1 

	proportions = np.zeros((num_clusters,num_colors)) 
	proportions_normalized = np.zeros((num_clusters,num_colors)) 
	for cluster in range(num_clusters):
		for color in range(num_colors):
			proportions[cluster,color] = np.dot(x[:,cluster],prob_vecs[:,color])

		proportions_normalized[cluster,:] = proportions[cluster,:]/div_total[cluster]

	return proportions_normalized, proportions, np.sum(x,axis=0)

# find  maxViol_from_proprtion 
def maxViol_from_proprtion(alpha, beta, num_clusters, proportions, sizes):
	gamma_fair =0 

	for counter in range(num_clusters):
		upper_viol = proportions[counter]-alpha*sizes[counter]
		lower_viol = beta*sizes[counter]-proportions[counter]
		max_viol = max(upper_viol,lower_viol)
		if max_viol>gamma_fair:
			gamma_fair = max_viol

	return gamma_fair 


# find  maxViol_from_proprtion, multi color. alpha and beta are arrays 
def maxViol_from_proprtion_multi_color(alpha, beta, num_colors, num_clusters, proportions, sizes):
	gamma_fair =0 

	for col in range(num_colors):
		for cluster_idx in range(num_clusters):
			upper_viol = proportions[cluster_idx,col]-alpha[col]*sizes[cluster_idx]
			lower_viol = beta[col]*sizes[cluster_idx]-proportions[cluster_idx,col]

			max_viol = max(upper_viol,lower_viol)
			if max_viol>gamma_fair:
				gamma_fair = max_viol

	return gamma_fair 


# find  maxViol_from_proprtion, multi color. alpha and beta are arrays 
def maxViol_Normalized_from_proprtion_multi_color(alpha, beta, num_colors, num_clusters, proportions, sizes):
	gamma_normalized = 0 

	for col in range(num_colors):
		for cluster_idx in range(num_clusters):
			upper_viol = proportions[cluster_idx,col]-alpha[col]*sizes[cluster_idx]
			lower_viol = beta[col]*sizes[cluster_idx]-proportions[cluster_idx,col]

			max_viol_norm = max(upper_viol,lower_viol)/sizes[cluster_idx] 

			if max_viol_norm>gamma_normalized:
				gamma_normalized = max_viol_norm

	return gamma_normalized 

# find  maxViol_from_proprtion, multi color. alpha and beta are arrays 
def maxRatioViol_from_proprtion_multi_color(alpha, beta, num_colors, num_clusters, proportions, sizes):
	gamma_fair =0 


	for col__ in range(num_colors):
		for cluster_index in range(num_clusters):
			if sizes[cluster_index] != 0 :
				upper_viol = proportions[cluster_index,col__]-alpha[col__]*sizes[cluster_index]
				lower_viol = beta[col__]*sizes[cluster_index]-proportions[cluster_index,col__]
				max_viol = max(upper_viol,lower_viol)
				denominator = 1*sizes[cluster_index]
				max_viol_prop = max_viol/denominator
	
				if max_viol_prop>gamma_fair:
					gamma_fair = max_viol_prop

	return gamma_fair 


# find max_Viol multi_color 
def max_RatioViol_multi_color(x,num_colors,prob_vecs,num_clusters,alpha,beta):
	_ , proportions, sizes = find_proprtions_multi_color(x,num_colors,prob_vecs,num_clusters)
	return maxRatioViol_from_proprtion_multi_color(alpha, beta, num_colors, num_clusters, proportions, sizes)




# find max_Viol
def max_Viol(x,num_colors,color_prob,num_clusters,alpha,beta):
	_ , proportions, sizes = find_proprtions_two_color(x,num_colors,color_prob,num_clusters)
	return maxViol_from_proprtion(alpha, beta, num_clusters, proportions, sizes)


# find max_Viol multi_color 
def max_Viol_multi_color(x,num_colors,prob_vecs,num_clusters,alpha,beta):
	_ , proportions, sizes = find_proprtions_multi_color(x,num_colors,prob_vecs,num_clusters)
	return maxViol_from_proprtion_multi_color(alpha, beta, num_colors, num_clusters, proportions, sizes)

# 
def max_Viol_Normalized_multi_color(x,num_colors,prob_vecs,num_clusters,alpha,beta):
	_ , proportions, sizes = find_proprtions_multi_color(x,num_colors,prob_vecs,num_clusters)
	return maxViol_Normalized_from_proprtion_multi_color(alpha, beta, num_colors, num_clusters, proportions, sizes)





def find_balance(x,num_colors, num_clusters,color_prob,proportion_data_set):
	proportions_normalized, _ , sizes = find_proprtions_two_color(x,num_colors,color_prob,num_clusters)
	min_balance = 1 
	balance_unfair = np.zeros(num_clusters)

	x = np.reshape(x , (-1,num_clusters)) 

	for i in range(num_clusters):
		if sizes[i]!=0:
			balance = min(proportions_normalized[i]/proportion_data_set , proportion_data_set/proportions_normalized[i])
			if (proportions_normalized[i]==0) or (proportion_data_set==0):
				pass 
		else: 
			balance = 10 

		balance_unfair[i] = balance
		if min_balance>balance:
			min_balance = balance 

	return min_balance




# find balance multi color, proportion_data_set is an array 
def find_balance_multi_color(x,num_colors, num_clusters,color_prob,proportion_data_set):
	proportions_normalized, _ , sizes = find_proprtions_multi_color(x,num_colors,color_prob,num_clusters)
	min_balance = 1 
	balance_unfair = np.zeros(num_clusters)

	for clust in range(num_clusters):
		for col_ in range(num_colors):
			if sizes[clust]!=0:
				balance = min(proportions_normalized[clust,col_]/proportion_data_set[col_] , proportion_data_set[col_]/proportions_normalized[clust,col_])
			else: 
				balance = 10 

			balance_unfair[clust] = balance
			if min_balance>balance:
				min_balance = balance 

	return min_balance





# for assignment for color-blind 
def x_for_colorBlind(preds,num_clusters):
	x = np.zeros((len(preds),num_clusters)) 
	for idx,p in enumerate(preds): 
		x[idx,p] = 1 

	return x.ravel().tolist() 

	