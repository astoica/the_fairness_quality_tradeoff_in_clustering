import networkx as nx
import csv 
import numpy as np
import random
import copy
import math 
from sklearn.cluster import SpectralClustering, KMeans
''' This code implements the repeated FCBC algorithm (Algorithm 2) with the group egalitarian objective. 
See requirements.txt and the README file for the required packages to run the code.'''

from sklearn.metrics import pairwise_distances
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import pickle
import scipy
import ast
import time 
import itertools
from itertools import permutations
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import random 
import pandas as pd 
import numba
from numba import jit , njit
from math import log2
from sklearn import preprocessing
import sys 
import timeit 
import datetime 
import json 
from collections import defaultdict
from scipy.spatial import distance
from fair_clustering_util import fair_clustering_util
from fair_clustering_lex import fair_clustering_lex
from util.configutil import read_list
from util.utilhelpers import get_clust_sizes, max_Viol_multi_color, x_for_colorBlind, max_RatioViol_multi_color, find_proprtions_two_color_deter 
import time

import configparser
import sys
import timeit
from pathlib import Path

from util.clusteringutil import (clean_data, read_data, scale_data,
                                 subsample_data, take_by_key,
                                 vanilla_clustering, write_fairness_trial)

from util.probutil import form_class_prob_vector, sample_colors, create_prob_vecs, sample_colors_ml_model
from cplex_fair_assignment_lp_solver_util import fair_partial_assignment_util
from cplex_fair_assignment_lp_solver_lex import fair_partial_assignment_lex

from util.pof_util_lex import relax_alpha_viol, relax_beta_viol, get_color_with_min_proportion, relax_util_viol, alpha_beta_array, get_viol_value_two_color, get_util_value_two_color, get_lex_multi_color

if __name__ == "__main__":

    # CHANGE AS NEEDED:
    # ----------------
    which_data = 'bluebike_'
    # which_data = 'census1990_'
    # which_data = 'adult_'
    num_cluster = [2] 
    deltas = [0.001] 
    max_points = 100 # example run 
    # ----------------


    alpha_POF = 1.03 # ratio between U_POF and the color-blind cost  

    num_colors = 2

    # Set LowerBound = 0
    LowerBound = 0
    L = LowerBound
    #
    # alpha0: is the first POF  
    alpha0= 1.001


    # POF lower bound on size for rounding 
    L_POF = 1 

    # alphaend: is the last POF   
    alphaend= 1.001

    # 
    alpha_step = 0.01/2

    # set ml_model_flag=False, p_acc=1.0
    ml_model_flag = False
    p_acc = 1.0 

    ''' POF ''' 
    # flag two color util 
    two_color_util=True
    r = 2**10 # 2**7

    epsilon = 1/r


    # config_file = "config/example_2_color_config.ini"
    # config_file = "config/example_multi_color_config.ini"
    config_file = "config/dataset_configs.ini"

    config = configparser.ConfigParser(converters={'list': read_list})
    config.read(config_file)


    if which_data == 'bluebike_':
        # config_str = "creditcard_binary_marriage" # if len(sys.argv) == 1 else sys.argv[1]
        config_str = "bluebike"

        # Read variables
        data_dir = config[config_str].get("data_dir")
        data_dir = 'output'
        # dataset = config[config_str].get("dataset")
        dataset = "bluebike"

    elif which_data == 'census1990_': 
        # config_str = "creditcard_binary_marriage" # if len(sys.argv) == 1 else sys.argv[1]
        config_str = "census1990_ss_sex"

        # Read variables
        data_dir = config[config_str].get("data_dir")
        data_dir = 'output'
        # dataset = config[config_str].get("dataset")
        dataset = "census1990_ss_sex"

    elif which_data == 'adult_':


        # config_str = "creditcard_binary_marriage" # if len(sys.argv) == 1 else sys.argv[1]
        config_str = "adult"

        # Read variables
        data_dir = config[config_str].get("data_dir")
        data_dir = 'output'
        # dataset = config[config_str].get("dataset")
        dataset = "adult"

    clustering_config_file = config[config_str].get("config_file")

    # ready up for the loop 
    alpha = np.linspace(start=alpha0 , stop=alphaend, num=int(((alphaend-alpha0)/alpha_step))+1, endpoint=True)

    df = pd.DataFrame(columns=['num_clusters','POF','UtilValue','UtilLP','LP Iters','opt_index','epsilon','Epsilon_set_size','minclustsize','Run_Time']) # ,'MaxViolFair','MaxViolUnFair','Fair Balance','UnFair Balance','Run_Time','ColorBlindCost','FairCost'])
    iter_idx = 0 


    #
    initial_score_save = 0 
    pred_save = [] 
    cluster_centers_save = [] 
    
    if type(num_cluster) is list:
        num_cluster = num_cluster[0] 

    counter = 0 

    alpha_POF_values = np.linspace(1,1.5,51)

    alphas_keeptrack = {}
    betas_keeptrack = {}
    clustering_cost = {}
    group_egalitarian_obj = {}
    assignment_keeptrack = {}

    for alpha_POF in alpha_POF_values: 
        print("alpha_POF: ", alpha_POF)

        
        csv_file = config[dataset]["csv_file"]
        df = pd.read_csv(csv_file,sep=config[dataset]["separator"])

        if config["DEFAULT"].getboolean("describe"):
            print(df.describe())

        # Subsample data if needed
        if max_points and len(df) > max_points:
            df = df.head(max_points)
        df, _ = clean_data(df, config, dataset)

        # variable_of_interest (list[str]) : variables that we would like to collect statistics for
        variable_of_interest = config[dataset].getlist("fairness_variable")

        # NOTE: this code only handles one color per vertex 
        assert len(variable_of_interest) == 1 

        attributes, color_flag, prob_vecs, prob_thresh = {}, {}, {}, {}  
        for variable in variable_of_interest:
            colors = defaultdict(list)
            this_color_flag = [0] * len(df)
            
            condition_str = variable + "_conditions"
            bucket_conditions = config[dataset].getlist(condition_str)

            # For each row, if the row passes the bucket condition, 
            # then the row is added to that color class
            for i, row in df.iterrows():
                for bucket_idx, bucket in enumerate(bucket_conditions):
                    if eval(bucket)(row[variable]):
                        colors[bucket_idx].append(i)  # add the point to the list of its colors 
                        this_color_flag[i] = bucket_idx  # record the color for this given point  

            # NOTE: colors is a dict, this_color_flag is a list
            attributes[variable] = colors     
            color_flag[variable] = this_color_flag

            if ml_model_flag==False: 
                prob_vecs[variable] = create_prob_vecs(len(df),p_acc,len(colors),this_color_flag)
            else:
                ml_model_path = 'MLModels' + '/' + dataset 
                prob_vecs_path = ml_model_path + '_prob_vecs.npy'
                n = len(df)
                prob_vecs[variable] = np.load(prob_vecs_path)[0:n,:]

        # representation (dict[str -> dict[int -> float]]) : representation of each color compared to the whole dataset
        representation = {}

        for var in variable_of_interest:
            color_proportions = np.sum(prob_vecs[var],axis=0)/len(df)
            dict_ = {} 
            for j in range(color_proportions.shape[0]):
                dict_.update({j:color_proportions[j]})

            representation[var] = dict_ 

        ( _ , color_proprtions), = representation.items()

        selected_columns = config[dataset].getlist("columns")
        df = df[[col for col in selected_columns]]



        # NOTE: this code only handles one membership criterion 
        ( _ , fair_vals), = representation.items()

        # NOTE: this handles the case when a color is missing in the sampled vertices 
        num_colors = max(fair_vals.keys())+1


        # Scale data if desired
        scaling = config["DEFAULT"].getboolean("scaling")
        if scaling:
            df = scale_data(df)

        # Cluster the data -- using the objective specified by clustering_method
        clustering_method = config["DEFAULT"]["clustering_method"]

        if counter ==0:
            initial_score, pred, cluster_centers = vanilla_clustering(df, num_cluster, clustering_method)
            initial_score_save, pred_save, cluster_centers_save = initial_score, pred, cluster_centers
        else:
            initial_score, pred, cluster_centers = initial_score_save, pred_save, cluster_centers_save


        # sizes (list[int]) : sizes of clusters
        sizes = [0 for _ in range(num_cluster)]
        for p in pred:
            sizes[p] += 1

        # dataset_ratio : Ratios for colors in the dataset
        dataset_ratio = {}
        for attr, color_dict in attributes.items():
            dataset_ratio[attr] = {int(color) : len(points_in_color) / len(df) 
                                for color, points_in_color in color_dict.items()}

        # fairness_vars (list[str]) : Variables to perform fairness balancing on
        fairness_vars = config[dataset].getlist("fairness_variable")

        # NOTE: here is where you set the upper and lower bounds 
        # NOTE: accross all different values within the same attribute you have the same multipliers up and down 
        for delta in deltas:
            alpha, beta = {}, {}
            if two_color_util:
                a_val, b_val = 1 + delta, 1 - delta
            else:

                a_val, b_val = 1 / (1 - delta) , 1 - delta

            for var, bucket_dict in attributes.items():
                alpha[var] = {k : a_val * representation[var][k] for k in bucket_dict.keys()}
                beta[var] = {k : b_val * representation[var][k] for k in bucket_dict.keys()}




            fp_color_flag, fp_alpha, fp_beta = (take_by_key(color_flag, fairness_vars),
                                                take_by_key(alpha, fairness_vars),
                                                take_by_key(beta, fairness_vars))




            alpha_orig= copy.deepcopy(fp_alpha)
            beta_orig = copy.deepcopy(fp_beta)

            print("alpha, beta: ", fp_alpha, fp_beta)
            alphas_keeptrack[alpha_POF] = copy.deepcopy(fp_alpha)
            betas_keeptrack[alpha_POF] = copy.deepcopy(fp_beta)

            # Solves partial assignment and then performs rounding to get integral assignment
            t1 = time.monotonic()
            res, nf_time, delta_lex, H_fixed = fair_partial_assignment_lex(df, cluster_centers, initial_score, delta, color_proprtions, fp_alpha, fp_beta, fp_color_flag, clustering_method, num_colors, L, epsilon,alpha_POF, L_POF)

            t2 = time.monotonic()
            lp_time = t2 - t1

            # print("alpha, beta: ", fp_alpha, fp_beta)

            # delta_lex_arr is an array, the index indicates the lex level and value in the objective 
            delta_lex_arr = np.zeros(num_colors) 
            for k,v in delta_lex.items():
                delta_lex_arr[k] = v 

            color_objectives_lp = np.zeros(num_colors) 

            for k, col_set in H_fixed.items():
                for col in col_set:
                    color_objectives_lp[col] = delta_lex[k]

                

            ### Output 
            output = {}

            # num_clusters for re-running trial
            output["num_clusters"] = num_cluster

            # Whether or not the LP found a solution
            output["partial_success"] = res["partial_success"]

            # Nonzero status -> error occurred
            output["partial_success"] = res["partial_success"]
            
            output["dataset_distribution"] = dataset_ratio

            # Save alphas and betas from trials
            output['prob_proportions'] = representation
            output["alpha"] = alpha
            output["beta"] = beta

            # Save original clustering score
            output["unfair_score"] = initial_score

            # Original Color Blind Assignments 
            #output["unfair_assignments"] = pred.tolist()

            # Clustering score after addition of fairness
            output["objective"] = res["objective"]
            
            # Clustering score after initial LP
            output["partial_fair_score"] = res["partial_objective"]

            # Save size of each cluster
            output["sizes"] = sizes

            output["attributes"] = attributes


            # These included at end because their data is large
            # Save points, colors for re-running trial
            # Partial assignments -- list bc. ndarray not serializable
            output["centers"] = [list(center) for center in cluster_centers]
            output["points"] = [list(point) for point in df.values]
            output["assignment"] = res["assignment"]

            output["partial_assignment"] = res["partial_assignment"]

            output["name"] = dataset
            output["clustering_method"] = clustering_method
            output["scaling"] = scaling
            output["delta"] = delta
            output["time"] = lp_time

            # NOTE: record proprtions
            output['partial_proportions'] = res['partial_proportions'] 
            output['proportions'] = res['proportions'] 

            output['partial_proportions_normalized'] = res['partial_proportions_normalized'] 
            output['proportions_normalized'] = res['proportions_normalized'] 

            # Record Lower Bound L
            output['Cluster_Size_Lower_Bound'] = L

            # Record Classifier Accurecy 
            output['p_acc'] = p_acc

            # 
            output['nf_time'] = nf_time 

            # Record probability vecs
            for k,v in prob_vecs.items():
                prob_vecs = v 

            output['prob_vecs'] = prob_vecs.ravel().tolist()

            # Writes the data in `output` to a file in data_dir
            # write_fairness_trial(output, data_dir)

            # Added because sometimes the LP for the next iteration solves so 
            # fast that `write_fairness_trial` cannot write to disk
            time.sleep(1) 

    
            color_objectives_rounded = get_lex_multi_color(np.reshape(output['proportions_normalized'], (num_colors,-1)),alpha_orig,beta_orig,num_colors)

            ( _ , color_flag), = color_flag.items()



            rounded_prop, _ , dummy = find_proprtions_two_color_deter(np.reshape(res["assignment"], (-1,num_cluster)) ,2,color_flag,num_cluster)
            lp_prop, _ , dummy = find_proprtions_two_color_deter(np.reshape(res["partial_assignment"], (-1,num_cluster)) ,2,color_flag,num_cluster)



            output['delta_lex'] = delta_lex_arr.tolist()

            output['color_objectives_lp'] = color_objectives_lp.tolist()

            output['color_objectives_rounded'] = color_objectives_rounded.tolist()

            output["epsilon"] = epsilon
            output["epsilon set size "] = 1/epsilon
            output["alpha_pof"] = alpha_POF


        clustering_cost[alpha_POF] = res['objective']**2
        group_egalitarian_obj[alpha_POF] = output['color_objectives_rounded'] 

    group_egalitarian_obj_max = {} 
    for key in group_egalitarian_obj.keys():
        group_egalitarian_obj_max[key] = max(group_egalitarian_obj[key])

    # Writing dictionary to a JSON file
    with open('output/' + which_data + 'clustering_cost' + str(max_points) + '_numclusters'+ str(num_cluster) + '_groupegalit.json', 'w') as f:
        json.dump(clustering_cost, f)


    # Writing dictionary to a JSON file
    with open('output/' + which_data + 'group_egalit_obj' + str(max_points) + '_numclusters'+ str(num_cluster) + '.json', 'w') as f:
        json.dump(group_egalitarian_obj, f)

