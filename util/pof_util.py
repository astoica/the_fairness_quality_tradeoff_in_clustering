''' This code is adapted from https://github.com/Seyed2357/Fair-Clustering-Under-Bounded-Cost and used for the repeated FCBC algorithm (Algorithm 2).'''

import numpy as np 
import random 

tolerance = 0.000001 


def alpha_beta_array(alpha):
	alpha_arr = np.zeros(2)
	for key, value in alpha.items():
		for k,v in value.items():
			alpha_arr[k] = v 

	return alpha_arr 



def get_viol_value_two_color(proportions,alpha,beta):
	alpha_ = alpha_beta_array(alpha)
	beta_  = alpha_beta_array(beta)
	max_prop = np.max(proportions,axis=1)
	min_prop = np.min(proportions,axis=1)
	zeros2 = np.zeros(2)

	viol_upper = np.maximum(max_prop-alpha_ , zeros2)
	viol_lower = np.maximum( beta_ -min_prop, zeros2) 
	return viol_upper,viol_lower


def get_util_value_two_color(proportions,alpha,beta):
	alpha_ = alpha_beta_array(alpha)
	beta_  = alpha_beta_array(beta)
	max_prop = np.max(proportions,axis=1)
	min_prop = np.min(proportions,axis=1)
	zeros2 = np.zeros(2)

	viol_upper = np.maximum(max_prop-alpha_ , zeros2)
	viol_lower = np.maximum( beta_ -min_prop, zeros2) 
	viol = np.maximum(viol_upper, viol_lower) 
	util_val = np.sum(viol)
	return util_val

def relax_alpha_viol(alpha,viol):
	alpha_return = alpha.copy()
	for key, value in alpha_return.items():
		for k,v in value.items():
			value[k] = v+viol

	return alpha_return 

def relax_beta_viol(beta,viol):
	beta_return = beta.copy()
	for key, value in beta_return.items():
		for k,v in value.items():
			value[k] = v-viol

	return beta_return 



#####
def relax_util_viol(beta_min,beta_max,alpha_min,alpha_max,alpha,beta,col_min,col_max,delta,r_min,r_max,viol):
	alpha_return = alpha.copy()
	beta_return = beta.copy()


	bound = delta*(r_max-r_min)
	print('\n\n BOUND')
	print(bound)

	for key, value in alpha_return.items():
		print(key)
		print(value)

		value[col_min] = min( alpha_min + (viol) , 1-tolerance) 
		if viol >= bound: 
			value[col_max] = min( alpha_max + (viol - bound) , 1-tolerance )
	

	for key, value in beta_return.items():
		print(key)
		print(value)

		value[col_min] = max( beta_min - (viol) , 0+tolerance ) 
		if viol >= bound: 
			value[col_max] = max( beta_max - (viol - bound) , 0+tolerance ) 

		print(alpha_return)
		print(beta_return)

	return alpha_return, beta_return






def get_color_with_min_proportion(color_proportions):
	r_min = 100 
	r_max = -1 
	col_min = -1 
	col_max = -1 

	for k,v in color_proportions.items():
		if v<r_min:
			col_min = k 
			r_min = v 
		if v>r_max:
			col_max = k 
			r_max = v 

	return r_min, col_min, r_max, col_max 


