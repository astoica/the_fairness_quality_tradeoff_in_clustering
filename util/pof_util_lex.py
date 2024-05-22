''' This code is adapted from https://github.com/Seyed2357/Fair-Clustering-Under-Bounded-Cost and used for the repeated FCBC algorithm (Algorithm 2).'''

import numpy as np 
import random 
import copy 

tolerance = 0.000001 


def look_for_color_in_H_fixed(H_fixed, color):
	for k,v in H_fixed.items():
		if color in v:
			return k 

def H_fixed_one_set(H_fixed):
	H_fixed_one = set() 
	for k,v in H_fixed.items():
		H_fixed_one = H_fixed_one.union(v)

	return H_fixed_one

def alpha_beta_array(alpha):
	alpha_arr = np.zeros(2)
	for key, value in alpha.items():
		for k,v in value.items():
			alpha_arr[k] = v 

	return alpha_arr 


def alpha_beta_array_multi_color(alpha,num_colors):
	alpha_arr = np.zeros(num_colors)
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


def get_lex_multi_color(proportions,alpha,beta,num_colors):
	alpha_ = alpha_beta_array_multi_color(alpha,num_colors)
	beta_  = alpha_beta_array_multi_color(beta,num_colors)
	max_prop = np.max(proportions,axis=1)
	min_prop = np.min(proportions,axis=1)
	zeros_ = np.zeros(num_colors)

	#print('\n\n\n\n\nCHECK ISSUE')
	#print(proportions.shape)
	#print(proportions)
	#print(max_prop.shape)
	#print(max_prop)
	#print(alpha_) 
	#print(zeros_)
	viol_upper = np.maximum(max_prop-alpha_ , zeros_)
	viol_lower = np.maximum( beta_ -min_prop, zeros_) 
	viol = np.maximum(viol_upper, viol_lower) 
	#print(viol_upper)
	#print(viol_lower)
	#print(viol)

	return viol









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



#####
'''
relax_lex_viol: 
1-Each color in H_fixed has its proportions unchanged.
2-Other colors are pushed by viol 
'''
def relax_lex_viol(alpha_orig, beta_orig, H_fixed, delta_lex, viol):
	# get the list of all non-chaning colors 
	H_fixed_one = H_fixed_one_set(H_fixed)

	alpha_relax = copy.deepcopy(alpha_orig)
	beta_relax = copy.deepcopy(beta_orig)

	#print('\n\n\n\n\n\n\n1------relax_lex_viol')
	#print(alpha_relax)
	#print(beta_relax)
	#print(viol)


	for key, value in alpha_relax.items():
		#print(key)
		#print(value)

		for k,v in value.items(): 
			#print(k)
			#print(v)
			#print(value[k])
			if k in H_fixed_one: 
				#print('actually here')
				#print(k)
				#print(H_fixed_one)
				value[k] = min( value[k]+delta_lex[look_for_color_in_H_fixed(H_fixed,k)]  , 1-tolerance ) 
			else:
				value[k] = min( value[k]+viol  , 1-tolerance )
	

	for key, value in beta_relax.items():
		#print(key)
		#print(value)

		for k,v in value.items(): 
			if k in H_fixed_one: 
			 	value[k] = max( value[k]-delta_lex[look_for_color_in_H_fixed(H_fixed,k)], 0+tolerance ) 
			else:
				value[k] = max( value[k]-viol  , 0+tolerance )



	#print('\n\n\n\n\n\n\n2------relax_lex_viol')	
	#print(alpha_relax)
	#print(beta_relax)

	return alpha_relax, beta_relax



## when all elements are in H_fixed 
def relax_lex_viol_all_fixed(alpha_orig, beta_orig, H_fixed, delta_lex):
	# get the list of all non-chaning colors 
	H_fixed_one = H_fixed_one_set(H_fixed)

	alpha_relax = copy.deepcopy(alpha_orig)
	beta_relax = copy.deepcopy(beta_orig)

	print('\n\n\n\n\n\n\n1------relax_lex_viol')
	print(alpha_relax)
	print(beta_relax)


	for key, value in alpha_relax.items():
		print(key)
		print(value)

		for k,v in value.items(): 
			print(k)
			print(v)
			print(value[k])
			if k in H_fixed_one: 
				print('actually here')
				print(k)
				print(H_fixed_one)
				value[k] = min( value[k]+delta_lex[look_for_color_in_H_fixed(H_fixed,k)]  , 1-tolerance ) 
			else:
				assert 1==0
				value[k] = min( value[k]  , 1-tolerance )
	

	for key, value in beta_relax.items():
		print(key)
		print(value)

		for k,v in value.items(): 
			if k in H_fixed_one: 
			 	value[k] = max( value[k]-delta_lex[look_for_color_in_H_fixed(H_fixed,k)], 0+tolerance ) 
			else:
				assert 1==0
				value[k] = max( value[k]  , 0+tolerance )



	print('\n\n\n\n\n\n\n2------relax_lex_viol')	
	print(alpha_relax)
	print(beta_relax)

	return alpha_relax, beta_relax



#####
'''
relax_lex_viol_possible: 
1-Each color in H_fixed has its proportions unchanged.
2-All colors except for the given color are fixed at the given viol. 
3-This color should move down (imrpove) by -round_POF-epsilon. 
4-You will send viol as an argument not viol improved. 
'''
def relax_lex_viol_possible(H_all,H_fixed,delta_lex,alpha_orig,beta_orig,viol,h_color,round_POF,epsilon):
	# h_color is the color that will move 
	H_fixed_one = H_fixed_one_set(H_fixed)

	# H_not_move is the collection of colors at viol  
	fixed_and_movable = H_fixed_one.union(set([h_color]))
	H_not_move = H_all.difference(fixed_and_movable)

	''' 
	print('\n-----------\n')
	print('CHECK relax_lex_viol_possible')
	print('H_fixed_one')
	print(H_fixed_one)
	print('h_color')
	print(h_color)
	print('fixed_and_movable')
	print(fixed_and_movable)
	print('H_not_move')
	print(H_not_move)
	''' 

	alpha_relax = copy.deepcopy(alpha_orig)
	beta_relax = copy.deepcopy(beta_orig)

	push_movable = max(viol-round_POF-epsilon, 0) 


	for key, value in alpha_relax.items():
		#print(key)
		#print(value)

		for k,v in value.items(): 
			#print(k)
			#print(v)
			#print(value[k])
			if k in H_fixed_one: 
				print('in H_fixed_one')
				print(k)
				value[k] = min( value[k]+delta_lex[look_for_color_in_H_fixed(H_fixed,k)]  , 1-tolerance ) 
			elif k in H_not_move:
				print('in H_not_move')
				print(k)
				value[k] = min( value[k]+viol  , 1-tolerance )
			else:
				print('in niether')
				print(k)
				value[k] = min( value[k]+push_movable  , 1-tolerance )



	for key, value in beta_relax.items():
		#print(key)
		#print(value)

		for k,v in value.items(): 
			#print(k)
			#print(v)
			#print(value[k])
			if k in H_fixed_one: 
			 	value[k] = max( value[k]-delta_lex[look_for_color_in_H_fixed(H_fixed,k)], 0+tolerance ) 
			elif k in H_not_move:
				value[k] = max( value[k]-viol  , 0+tolerance )
			else:
				value[k] = max( value[k]-push_movable  , 0+tolerance )

	return alpha_relax, beta_relax





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


