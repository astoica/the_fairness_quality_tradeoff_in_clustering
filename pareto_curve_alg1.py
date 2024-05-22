''' This file contains the code for running Algorithm 1.
See requirements.txt for the needed requirements to run the code.'''
import numpy as np
import copy
from sklearn.cluster import KMeans
import pickle
import pandas as pd 
from math import log2
from collections import defaultdict
from os.path import exists

from dynprogr_fcts import sort_by_cost_fairness, compute_fairness_all_clusterings, compute_list_of_all_partitions_opt2, compute_T_list_nomovingcenters_memoryopt, get_lex_sum_multi_color

from nf_rounding_multi_color import min_cost_rounding_multi_color, find_proprtions

from fair_clustering_util import fair_clustering_util
from util.configutil import read_list
from util.utilhelpers import get_clust_sizes, max_Viol_multi_color, x_for_colorBlind, max_RatioViol_multi_color, find_proprtions_two_color_deter 
import configparser
from pathlib import Path
from util.clusteringutil import (clean_data, read_data, scale_data,
                                 subsample_data, take_by_key,
                                 vanilla_clustering, write_fairness_trial)

from util.probutil import form_class_prob_vector, sample_colors, create_prob_vecs, sample_colors_ml_model
from cplex_fair_assignment_lp_solver_util import fair_partial_assignment_util
from util.pof_util import relax_alpha_viol, relax_beta_viol, get_color_with_min_proportion, relax_util_viol, alpha_beta_array, get_viol_value_two_color, get_util_value_two_color



if __name__ == "__main__":

    # CHANGE AS NEEDED:
    # ----------------
    which_data = 'bluebike_'
    # which_data = 'census1990_'
    # which_data = 'adult_'

    # fairness_metric = 'balance'
    # fairness_metric = 'sumofimbalances'
    fairness_metric = 'group-util'
    # fairness_metric = 'group-util-sum'
    # fairness_metric = 'group-egalit'
    # fairness_metric = 'group-egalit-sum'

    num_cluster = [2] # This is the number of clusters 
    deltas = [0.001] # This is the value of the delta, the deviation allowed. 
    max_points = 100 # This is the size of the subsampled data, setting 100 as an example for the reader.
    # ----------------


    # SETUP
    num_colors = 2

    # Set LowerBound = 0
    LowerBound = 0
    L = LowerBound
    #
    # alpha0: is the first POF  
    alpha0 = 1.001
    # alphaend: is the last POF   
    alphaend= 1.001
    alpha_step = 0.01/2

    # set ml_model_flag=False, p_acc=1.0
    ml_model_flag = False
    p_acc = 1.0 

    ''' POF ''' 
    # flag two color util 
    two_color_util=True
    r = 2**10 # 2**7

    epsilon = 1/r

    config_file = "config/dataset_configs.ini"

    config = configparser.ConfigParser(converters={'list': read_list})
    config.read(config_file)


    subsample_flag = 'no'

    if fairness_metric == 'sumofimbalances':
        subsample_flag = 'yes'
        
    if which_data == 'bluebike_':
        # config_str = "creditcard_binary_marriage" # if len(sys.argv) == 1 else sys.argv[1]
        config_str = "bluebike"

        # Read variables
        data_dir = config[config_str].get("data_dir")
        data_dir = 'output'
        # dataset = config[config_str].get("dataset")
        dataset = "bluebike"
        clustering_config_file = config[config_str].get("config_file")
        df_all = pd.read_csv('data/bluebikedata_201501_binarygender.csv')

    elif which_data == 'census1990_':

        # config_str = "creditcard_binary_marriage" # if len(sys.argv) == 1 else sys.argv[1]
        config_str = "census1990_ss_sex"

        # Read variables
        data_dir = config[config_str].get("data_dir")
        data_dir = 'output'
        # dataset = config[config_str].get("dataset")
        dataset = "census1990_ss_sex"
        clustering_config_file = config[config_str].get("config_file")
        df_all = pd.read_csv('data/subsampled_census1990.csv')

    elif which_data == 'adult_': 

        # config_str = "creditcard_binary_marriage" # if len(sys.argv) == 1 else sys.argv[1]
        config_str = "adult"

        # Read variables
        data_dir = config[config_str].get("data_dir")
        data_dir = 'output'
        # dataset = config[config_str].get("dataset")
        dataset = "adult"
        clustering_config_file = config[config_str].get("config_file")
        df_all = pd.read_csv('data/adult.csv')
        

    alpha = np.linspace(start=alpha0 , stop=alphaend, num=int(((alphaend-alpha0)/alpha_step))+1, endpoint=True)

    iter_idx = 0 

    initial_score_save = 0 
    pred_save = [] 
    cluster_centers_save = [] 
    
    if type(num_cluster) is list:
        num_cluster = num_cluster[0] 

    counter = 0 

    list_nodes = list(range(max_points))
    n = len(list_nodes)
    k = num_cluster

    csv_file = config[dataset]["csv_file"]
    df0 = pd.read_csv(csv_file,sep=config[dataset]["separator"])

    if config["DEFAULT"].getboolean("describe"):
        print(df0.describe())

    # Subsample data if needed (e.g. for the sum of imbalances objective)
    if subsample_flag == 'yes':
        if which_data == 'adult_':
            # pick the same number of red and blue points!
            filtered_df_1 = df0[df0['sex'] == ' Male']
            filtered_df_2 = df0[df0['sex'] == ' Female']

        elif which_data == 'bluebike_':
            # pick the same number of red and blue points!
            filtered_df_1 = df0[df0['gender'] == 1]
            filtered_df_2 = df0[df0['gender'] == 2]

        elif which_data == 'census1990_':
            # pick the same number of red and blue points!
            filtered_df_1 = df0[df0['iSex'] == 1]
            filtered_df_2 = df0[df0['iSex'] == 0]

        # pick the same number of red and blue points!
        #filtered_df_1 = df0[df0['gender'] == 1]
        # Sample n rows from the filtered DataFrame
        sampled_df_1 = filtered_df_1.sample(n=int(max_points/2))
        #filtered_df_2 = df0[df0['gender'] == 2]
        # Sample n rows from the filtered DataFrame
        sampled_df_2 = filtered_df_2.sample(n=int(max_points/2))
        df = pd.concat([sampled_df_1, sampled_df_2], axis=0 , ignore_index=True)

    else: 
        # Subsample data if needed                                                                                                                                                                                                                         
        if max_points and len(df0) > max_points:
            df = df0.head(max_points)

    df_retain = df.copy()

    df, _ = clean_data(df, config, dataset)

    # variable_of_interest (list[str]) : sensitive attribute variables
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
        #   alpha_i = a_val * (representation of color i in dataset)
        #   beta_i  = b_val * (representation of color i in dataset)
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

    ( _ , color_flag), = color_flag.items()

    # build the c dictionary, a dictionary with keys as datapoint indices and values the sensitive attribute value 
    # (for two sensitive attributes, 1 is the majority and 0 is the minority)

    if which_data == 'bluebike_':
        c = {}
        for i in range(max_points):
            if df_retain.iloc[i]['gender'] == 1:
                c[i] = 1
            else:
                c[i] = 0
    elif which_data == 'census1990_':
        c = {}
        for i in range(max_points):
            if df_retain.iloc[i][variable_of_interest][0] == 1:
                c[i] = 1
            else:
                c[i] = 0

    elif which_data == 'adult_':
        c = {}
        for i in range(max_points):
            if df_retain.iloc[i]['sex'] == ' Male':
                c[i] = 1
            else:
                c[i] = 0

    U = df.to_numpy()

    c_list = np.array(list(c.values()))


    # performing manual spectral clustering: using the first k values of the eigenspace to do k-means 
    #km = KMeans(init='k-means++', n_clusters=k, max_iter=200, n_init=200, verbose=0, random_state=3425)
    km = KMeans(init='k-means++', n_clusters=k, max_iter=200, n_init=200, verbose=0)

    km.fit(df)
    y = km.labels_
    # distances from the k-centers
    X_dist = km.transform(df)**2

    original_centers = km.cluster_centers_ 
    initial_centers = original_centers 


    no_of_blue = 0
    no_of_red = 0 
    for i in range(len(list_nodes)):
        if c[i] == 1:
            no_of_blue += 1
        else:
            no_of_red += 1


    orderings = {}
    number_of_orderings = 1 # this is the number of orderings of the dataset 
    for ord in range(number_of_orderings):
        orderings[ord] = list(range(0,len(list_nodes)))
        orderings[ord] = np.array(orderings[ord])

    list_of_partitions_ordering_opt2 = {}
    for ord in range(number_of_orderings):
        list_of_partitions_ordering_opt2[ord] = compute_list_of_all_partitions_opt2(orderings[ord], k, c)

    T_ordering_opt2_memory = {}
    for ord in range(number_of_orderings):
        T_ordering_opt2_memory[ord] = compute_T_list_nomovingcenters_memoryopt(list(orderings[ord]), U, k, c, initial_centers, list_of_partitions_ordering_opt2[ord], X_dist)

    quality_fairness_notmoving_fixedeval_optmem, quality_fairness_notmoving_movingeval_optmem, dict_costfairness_clustering_optmem = compute_fairness_all_clusterings(df, list_nodes, number_of_orderings, fairness_metric, list_of_partitions_ordering_opt2, T_ordering_opt2_memory, U, k, initial_centers, c, num_colors, color_flag, num_cluster, alpha_orig, beta_orig, var)

    allpoints_fixedeval, allpoints_movingeval, paretofront_fixedeval, paretofront_movingeval = sort_by_cost_fairness(number_of_orderings, fairness_metric, quality_fairness_notmoving_fixedeval_optmem, quality_fairness_notmoving_movingeval_optmem)

    # paretofront_fixedeval computes all the points on the Pareto front with cost and fairness values, where the cost is computed with respect to the centroids found by a vanilla clustering algorithm
    with open('output/' + which_data + 'top'+  str(max_points) + 'datapoints_' + fairness_metric + '_numclusters' + str(num_cluster) + '_paretofrontfixedeval'+ '.pickle', 'wb') as file:
        pickle.dump(paretofront_fixedeval, file)

    # paretofront_fixedeval computes all the points on the Pareto front with cost and fairness values, where the cost is computed with respect to the recomputed centroids based on the current clustering
    with open('output/' + which_data + 'top'+  str(max_points) + 'datapoints_' + fairness_metric + '_numclusters' + str(num_cluster) +'_paretofrontmovingeval'+ '.pickle', 'wb') as file:
        pickle.dump(paretofront_movingeval, file)
