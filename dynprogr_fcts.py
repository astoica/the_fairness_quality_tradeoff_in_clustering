''' This file contains the code for all auxiliary functions needed to run Algorithm 1, which is stored in pareto_curve_alg1.py.
See requirements.txt for the needed requirements to run the code.'''
import numpy as np
import copy
import ast
import time 
from collections import Counter
from nf_rounding_multi_color import min_cost_rounding_multi_color, find_proprtions
from util.pof_util_lex import relax_alpha_viol, relax_beta_viol, get_color_with_min_proportion, relax_util_viol, alpha_beta_array, get_viol_value_two_color, get_util_value_two_color, get_lex_multi_color, alpha_beta_array_multi_color
from fair_clustering_util import fair_clustering_util
from util.configutil import read_list
from util.utilhelpers import get_clust_sizes, max_Viol_multi_color, x_for_colorBlind, max_RatioViol_multi_color, find_proprtions_two_color_deter 
import time

''' AUXILIARY FUNCTIONS '''

def get_util_sum_value_two_color(proportions,alpha,beta):
    ''' This function computes the group utilitarian value for a particular proportion value.
    Args: proportions (array of proportions of the differennt groups in the clustering); 
    alpha, beta (arrays with the pre-specified proportions of each group in the clustering)
    Output: group utilitarian objecive value'''
    alpha_ = alpha_beta_array(alpha)
    beta_  = alpha_beta_array(beta)
    max_prop = np.max(proportions,axis=1)
    min_prop = np.min(proportions,axis=1)
    zeros2 = np.zeros(2)
    viol_upper = np.maximum(max_prop-alpha_ , zeros2)
    viol_lower = np.maximum( beta_ -min_prop, zeros2) 
    viol = viol_upper + viol_lower 
    util_val = np.sum(viol)
    return util_val

def get_lex_sum_multi_color(proportions,alpha,beta,num_colors):
    ''' This function computes the group egalitarian objective value for a particular proportion value.
    Args: proportions (array of proportions of the differennt groups in the clustering); 
    alpha, beta (arrays with the pre-specified proportions of each group in the clustering)
    Output: group utilitarian objecive value'''
    alpha_ = alpha_beta_array_multi_color(alpha,num_colors)
    beta_  = alpha_beta_array_multi_color(beta,num_colors)
    max_prop = np.max(proportions,axis=1)
    min_prop = np.min(proportions,axis=1)
    zeros_ = np.zeros(num_colors)
    viol_upper = np.maximum(max_prop-alpha_ , zeros_)
    viol_lower = np.maximum( beta_ -min_prop, zeros_) 
    viol = viol_upper + viol_lower 
    return viol


def compute_avg_kdistance_cl_contribution(my_cluster, km_distances,node_index):
    ''' This function computes the distance between a node with index node_index and the cluster center of cluster my_cluster. 
    Args: my_cluster is the index of the cluster of interest; km_distances is an array with entries the distances between 
    datapoints (by row) and the cluster centers (by column); node_index is the index of the datapoint of interest.'''
    return km_distances[node_index][my_cluster]

def cluster_proportions(cdict, list_nodes_G, no_clusters, cluster_assignment):
    ''' This function computes the proportions of points of each sensitive attribute in each cluster.
    Args: cdict is a dictionary with keys as the datapoints indices and values the sensitive attribute value (0 for majority, 1 for minority); 
    list_nodes_G is the list of datapoints indices; no_clusters denotes the number of clusters; cluster_assignment is a list denoting the cluster
    assignment of each datapoint.'''
    # sizes of clusters    
    count = Counter(cluster_assignment)
    cluster_sizes = {kk: count.get(kk, 0) for kk in range(no_clusters)}
    # cluster_proportion is a dictionary mapping from the clusters to the proportion of the majority nodes in each cluster 
    cluster_proportion = {}
    # cluster_majority is a dictionary mapping from the clusters to the number of the majority nodes in each cluster
    cluster_majority = {}
    # cluster_mminority is a dictionary mapping from the clusters to the number of the minority nodes in each cluster
    cluster_minority = {}
    for kk in range(no_clusters):
        cluster_proportion[kk] = 0
        cluster_majority[kk] = 0
        cluster_minority[kk] = 0

    for u in range(len(list_nodes_G)):
        if cdict[u] == 1:
            cluster_proportion[np.array(cluster_assignment)[u]] += 1
            cluster_majority[np.array(cluster_assignment)[u]] += 1
        else:
            cluster_minority[np.array(cluster_assignment)[u]] += 1

    for kk in range(no_clusters):
        if cluster_sizes[kk] == 0:
            cluster_proportion[kk] = 0
        else:
            cluster_proportion[kk] /= cluster_sizes[kk]
    return cluster_sizes, cluster_proportion, cluster_majority, cluster_minority

def compute_centroids_manually_opt(U_matrix, cluster_assignment, no_of_clusters, my_original_centers):
    ''' This function computes centroids of clusters; if there is an empty cluster, it uses the pre-specified centroids.
    Args: U_matrix is a matrix of features for the datapoints; cluster_assignment is a list denoting the cluster
    assignment of each datapoint; no_of_clusters is the number of clusters; my_original_centers is an array of 
    the initial centroids found by a vanilla clustering algorithm.
    Output: new_centers as an array of the computed centroids. '''
    cluster_assignment = np.asarray(cluster_assignment)
    U_matrix = np.asarray(U_matrix)
    
    # Initialize new_centers with the original centers
    new_centers = np.array(my_original_centers, copy=True)
    
    # Calculate the sum and count of points in each cluster
    for kk in range(no_of_clusters):
        indices = np.where(cluster_assignment == kk)[0]
        if indices.size > 0:
            new_centers[kk] = np.mean(U_matrix[indices], axis=0)
    
    return new_centers

def compute_km_distances_opt(points, centroids, no_of_clusters):
    ''' This function computes the Euclidean distance between each datapoint the centroids.
    Args: points is the array of datapoints; centroids is the array of centroids of clusters; 
    no_of_clusters is the number of clusters.
    Output: X_dist_new is the new array of distances between each datapoint and all the centroids 
    (datapoint by row, centroid by column).'''
    diff =  points[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    X_dist_new = np.sum(diff**2, axis=-1)

    return X_dist_new

def compute_avg_kdistance_inertia_opt(cluster_assignmentx, no_clusters, km_distances):
    ''' This function computes the average k-means distance between datapoints and their assigned cluster centers.
    Args: cluster_assignmentx is is a list denoting the cluster assignment of each datapoint; 
    no_clusters is the number of clusters; km_distances is an array of the distances between all datapoints (by row)
    and all cluster centroids (by column).'''
    sumdist = sum((km_distances[cluster_assignmentx == k, k].sum() for k in range(no_clusters)))
    return sumdist



''' FAIRNESS OBJECTIVES '''

def compute_sumofimbalances(c_list, yy, kk):
    ''' This function computes the linearized balance of a particular clustering assignment. 
    Args: c_list is list of datapoints sensitive attribute (1 for majority and 0 for minority); 
    cluster_assignment is a list denoting the cluster assignment of each datapoint; kk denotes 
    the number of clusters.
    Output: the sum of imbalances.'''
    # Calculate the total counts for each y value
    total_counts = np.bincount(yy, minlength=kk)
    
    # Calculate the counts where c == 1 for each y value
    count_c1_y = np.bincount(yy[c_list == 1], minlength=kk)
    
    # Calculate proportions
    proportions_unnorm = np.zeros(kk)
    non_zero_indices = total_counts != 0
    proportions_unnorm[non_zero_indices] = count_c1_y[non_zero_indices] 
    prop_min_unnorm = total_counts - proportions_unnorm

    return np.sum(np.abs(prop_min_unnorm - proportions_unnorm))

def compute_balance_minprop_opt(c_list, yy, kk):
    ''' This function computes the balance objective.
    Input: c_list is a list of sensitive attribute for each datapoint (for two attributes, 
    it's 1 for the majority and 0 for the minority); yy is a list denoting the cluster
    assignment of each datapoint; kk is the number of clusters.
    Output: mini_value is the value of the balance objective.'''
    # Calculate the total counts for each y value
    total_counts = np.bincount(yy, minlength=kk)
    
    count_c1_y = np.bincount(yy[c_list == 1], minlength=kk)
    
    proportions = np.zeros(kk)
    non_zero_indices = total_counts != 0
    proportions[non_zero_indices] = count_c1_y[non_zero_indices] / total_counts[non_zero_indices]
    
    mini_prop = np.minimum(proportions, 1 - proportions)
    mini_value = np.min(mini_prop)
    return mini_value

def reconstructed_grouputil_kclusters(df, yy_final, num_colors, color_flag, num_cluster, alpha_orig, beta_orig):
    ''' This function computes the group utilitarian objective for a clustering.
    Args: df is the dataframe of the features; yy_final is an array denoting the cluster
    assignment of each datapoint; num_colors is the number of sensitive attributes used; 
    color_flag is a variable denoting the sensitive attribute of each datapoint; num_cluster
    is the number of clusters; alpha_orig and beta_orig are the pre-specified proportions of 
    each sensitive attribute in a cluster.
    Output: util_objective_value is the value of the group utilitarian objective.'''
    identity_matrix = np.eye(num_cluster, dtype=int)

    y_prime = identity_matrix[np.array(yy_final)].flatten()
    y_prime = y_prime.astype(float)

    test_array = np.zeros((len(df),num_cluster))
    original_shape = test_array.shape
    x_rounded_reconstructed = np.array(y_prime).reshape(original_shape)

    rounded_proportions_normalized, rounded_proportions = find_proprtions(x_rounded_reconstructed,num_colors,color_flag,num_cluster)
    proportions_normalized = rounded_proportions_normalized.ravel().tolist()

    util_objective_value = get_util_value_two_color(np.reshape(proportions_normalized, (2,-1)),alpha_orig,beta_orig)
    return util_objective_value


def reconstructed_grouputilsum_kclusters(df, yy_final, num_colors, color_flag, num_cluster, alpha_orig, beta_orig):
    ''' This function computes the group utilitarian-sum objective for a clustering.
    Args: df is the dataframe of the features; yy_final is an array denoting the cluster
    assignment of each datapoint; num_colors is the number of sensitive attributes used; 
    color_flag is a variable denoting the sensitive attribute of each datapoint; num_cluster
    is the number of clusters; alpha_orig and beta_orig are the pre-specified proportions of 
    each sensitive attribute in a cluster.
    Output: util_objective_value is the value of the group utilitarian-sum objective.'''
    
    identity_matrix = np.eye(num_cluster, dtype=int)

    y_prime = identity_matrix[np.array(yy_final)].flatten()
    y_prime = y_prime.astype(float)


    test_array = np.zeros((len(df),num_cluster))
    original_shape = test_array.shape
    x_rounded_reconstructed = np.array(y_prime).reshape(original_shape)

    rounded_proportions_normalized, rounded_proportions = find_proprtions(x_rounded_reconstructed,num_colors,color_flag,num_cluster)
    proportions_normalized = rounded_proportions_normalized.ravel().tolist()

    util_objective_test = get_util_sum_value_two_color(np.reshape(proportions_normalized, (2,-1)),alpha_orig,beta_orig)
    return util_objective_test


def reconstructed_groupegalit_kclusters(df, yy_final, num_colors, color_flag, num_cluster, alpha_orig, beta_orig):
    ''' This function computes the group egalitarian objective for a clustering.
    Args: df is the dataframe of the features; yy_final is an array denoting the cluster
    assignment of each datapoint; num_colors is the number of sensitive attributes used; 
    color_flag is a variable denoting the sensitive attribute of each datapoint; num_cluster
    is the number of clusters; alpha_orig and beta_orig are the pre-specified proportions of 
    each sensitive attribute in a cluster.
    Output: util_objective_value is the value of the group egalitarian objective.'''
    identity_matrix = np.eye(num_cluster, dtype=int)

    y_prime = identity_matrix[np.array(yy_final)].flatten()
    y_prime = y_prime.astype(float)

    test_array = np.zeros((len(df),num_cluster))
    original_shape = test_array.shape
    x_rounded_reconstructed = np.array(y_prime).reshape(original_shape)

    rounded_proportions_normalized, rounded_proportions = find_proprtions(x_rounded_reconstructed,num_colors,color_flag,num_cluster)
    proportions_normalized = rounded_proportions_normalized.ravel().tolist()

    color_objectives_rounded = get_lex_multi_color(np.reshape(proportions_normalized, (num_colors,-1)),alpha_orig,beta_orig,num_colors)

    return max(color_objectives_rounded.tolist())

def reconstructed_groupegalitsum_kclusters(df, yy_final, num_colors, color_flag, num_cluster, alpha_orig, beta_orig):
    ''' This function computes the group egalitarian-sum objective for a clustering.
    Args: df is the dataframe of the features; yy_final is an array denoting the cluster
    assignment of each datapoint; num_colors is the number of sensitive attributes used; 
    color_flag is a variable denoting the sensitive attribute of each datapoint; num_cluster
    is the number of clusters; alpha_orig and beta_orig are the pre-specified proportions of 
    each sensitive attribute in a cluster.
    Output: util_objective_value is the value of the group egalitarian-sum objective.'''
    
    identity_matrix = np.eye(num_cluster, dtype=int)

    y_prime = identity_matrix[np.array(yy_final)].flatten()
    y_prime = y_prime.astype(float)


    test_array = np.zeros((len(df),num_cluster))
    original_shape = test_array.shape
    x_rounded_reconstructed = np.array(y_prime).reshape(original_shape)

    rounded_proportions_normalized, rounded_proportions = find_proprtions(x_rounded_reconstructed,num_colors,color_flag,num_cluster)
    proportions_normalized = rounded_proportions_normalized.ravel().tolist()

    color_objectives_rounded = get_lex_sum_multi_color(np.reshape(proportions_normalized, (num_colors,-1)),alpha_orig,beta_orig,num_colors)

    return max(color_objectives_rounded.tolist())


''' DYNAMIC PROGRAMMING TABLE FUNCTIONS '''

def compute_list_of_all_partitions_opt2(list_nodes_new_x, kc, cdict):
    ''' This function computes the list of all possible patters given a dataset, to be looped through in the dynamic programming approach.
    Args: list_nodes_new_x is the list of datapoints in some ordering; kc is the number of clusters; 
    cdict is a dictionary with keys as the datapoints and values as the sensitive attribute value (for two atttributes,
    1 denotes the majority and 0 denotes the minority). 
    Output: list_of_partitions_x as the list of possible patterns.'''
    list_of_partitions_x = {}
    initial_partition = tuple((0, 0) for _ in range(kc))
    list_of_partitions_x[0] = {initial_partition}

    for i in range(len(list_nodes_new_x)):
        current_partitions = list_of_partitions_x[i]
        new_partitions = set()

        for prev_partition in current_partitions:
            for j in range(kc):
                new_partition = list(prev_partition) 
                new_cluster = list(new_partition[j])  # Extract cluster to modify
                new_cluster[cdict[list_nodes_new_x[i]]] += 1  # Increment the count for either of the two sensitive attributes
                new_partition[j] = tuple(new_cluster)  # Reinsert modified cluster back as a tuple
                new_partitions.add(tuple(new_partition))  # Add the modified partition as a tuple of tuples

        # Store unique partitions only
        list_of_partitions_x[i + 1] = new_partitions

    return list_of_partitions_x

def compute_T_list_nomovingcenters_memoryopt(list_nodes_new_x, Ux, kc, cdict, initial_centers_x, list_of_partitions_x, X_dist):
    ''' This is Algorithm 1. This function computes the dynamic programming table T using the list of patterns from the data, 
    by finding the best clustering that minimizes cost.
    Args: list_nodes_new_x is a list of datapoints in some ordering; Ux is the matrix of features; kc is the number of clusters;
    cdict is a dictionary with keys as the datapoints and values as the sensitive attribute value (for two atttributes,
    1 denotes the majority and 0 denotes the minority); initial_centers_x is an array of original centroids found by a 
    vanilla clustering algorithm; list_of_partitions_x is a list of possible patterns of the dataset; X_dist is the array
    of distances between datapoints and original centroids found by a vanilla clustering algorithms (datapoint by row,
    centroid by column).
    Output: table Tx'''
    Tx = {}
    initial_key = tuple((0, 0) for _ in range(kc)) # this is for two sensitive attributes
    Tx[initial_key] = (0, [None] * len(list_nodes_new_x))

    for i in range(len(list_nodes_new_x)):
        new_Tx = {}
        for p in list_of_partitions_x[i + 1]:
            sp = {}
            for j in range(kc):
                if p[j][cdict[list_nodes_new_x[i]]] > 0:
                    p2 = [list(cluster) for cluster in p]  
                    p2[j][cdict[list_nodes_new_x[i]]] -= 1
                    p2_key = tuple(tuple(cluster) for cluster in p2)

                    if p2_key in Tx:
                        q, y = Tx[p2_key]
                        y_new = y[:] 
                        y_new[i] = j
                        # Recompute the cluster cost q for only the updated cluster
                        q_new = q + X_dist[i][j]

                        # Attempt to update or create a new entry in sp
                        new_tuple = (q_new, y_new)
                        if p2_key not in sp or sp[p2_key][0] > q_new:
                            sp[p2_key] = new_tuple

            # Minimize q and update new_Tx for the current pattern
            min_sp_key = min(sp, key=lambda x: sp[x][0])
            new_Tx[tuple(tuple(cluster) for cluster in p)] = sp[min_sp_key]

        Tx = new_Tx  # Replace Tx with new_Tx, dropping old patterns not needed

    return Tx


''' COMPUTING ALL POINTS IN THE PARETO SPACE AND FILTERING OUT DOMINATED POINTS '''

def compute_fairness_all_clusterings(df, list_nodes, number_of_orderings, fairness_metric, list_of_partitions_ordering, T_ordering, U, k, initial_centers, c, num_colors, color_flag, num_cluster, alpha_orig, beta_orig, var): 
    ''' This function computes the fairness objective for all possible clusterings presented by the table T_ordering. 
    Args: df the dataframe of features; list_nodes is the list of datapoints indices; number_of_orderings is the number 
    of orderings of datapoints used for computation; fairness_metric is a string denoting the fairness objective used;
    list_of_partitions_ordering is the list of patterns of the datapoints; T_ordering is the dynamic programming table T
    outputted by Algorithm 1; U is the array of features; k is the number of clusters; initial_centers is the array of 
    centroids found by a vanilla clustering algorithm; c is a dictionary with keys as the datapoints and values as the 
    sensitive attribute value (for two atttributes, 1 denotes the majority and 0 denotes the minority); num_colors is 
    the number of sensitive attributes used; color_flag is a variable indicating the sensitive attribute of datapoints; 
    num_cluster is the number of clusters; alpha_orig and beta_orig are the proportions inputted in the proportional
    violation-based fairness objectives; var is the variable of interest (e.g. gender, race, etc).'''

    quality_fairness_notmoving_fixedeval = {}
    quality_fairness_notmoving_movingeval = {}

    for ord in range(number_of_orderings):
        quality_fairness_notmoving_fixedeval[ord] = []
        quality_fairness_notmoving_movingeval[ord] = []
        

    costfixed = {} 
    costmoving = {}
    fairness_avg_all = {}
    fairness_avg = {}

    dict_costfairness_clustering = {} 

    c_list = np.array(list(c.values()))

    for ord in range(number_of_orderings):
        dict_costfairness_clustering[ord] = {}
        for p in list_of_partitions_ordering[ord][len(df)]:
            yy = T_ordering[ord][p][1]
            costfixed[ord] = T_ordering[ord][p][0]

            new_centers1 = compute_centroids_manually_opt(U, np.array(yy), k, initial_centers)
            new_X_dist1 = compute_km_distances_opt(U, new_centers1, k)
            costmoving[ord] = compute_avg_kdistance_inertia_opt(np.array(yy), k, new_X_dist1)

            if fairness_metric == 'balance':
                fairness_avg[ord] = compute_balance_minprop_opt(c_list, np.array(yy), k)
            elif fairness_metric == 'sumofimbalances':
                fairness_avg[ord] = compute_sumofimbalances(c_list, np.array(yy), k)
            elif fairness_metric == 'group-util':
                fairness_avg[ord] = reconstructed_grouputil_kclusters(df, yy, num_colors, color_flag, num_cluster, alpha_orig, beta_orig)
            elif fairness_metric == 'group-util-sum':
                fairness_avg[ord] = reconstructed_grouputilsum_kclusters(df, yy, num_colors, color_flag, num_cluster, alpha_orig, beta_orig)
            elif fairness_metric == 'group-egalit':
                fairness_avg[ord] = reconstructed_groupegalit_kclusters(df, yy, num_colors, color_flag, num_cluster, alpha_orig, beta_orig)
            elif fairness_metric == 'group-egalit-sum':
                fairness_avg[ord] = reconstructed_groupegalitsum_kclusters(df, yy, num_colors, color_flag, num_cluster, alpha_orig, beta_orig)

            quality_fairness_notmoving_fixedeval[ord].append(tuple([costfixed[ord], fairness_avg[ord]]))
            quality_fairness_notmoving_movingeval[ord].append(tuple([costmoving[ord], fairness_avg[ord]]))
            dict_costfairness_clustering[ord][(costmoving[ord], fairness_avg[ord])] = yy

    return quality_fairness_notmoving_fixedeval, quality_fairness_notmoving_movingeval, dict_costfairness_clustering

def sort_by_cost_fairness(number_of_orderings, fairness_metric, quality_fairness_notmoving_fixedeval, quality_fairness_notmoving_movingeval): 
    ''' This function filters out the dominated points in order to compute the Pareto front by sorting first by cost and then by fairness.
    Input: number_of_orderings is the number of orderings of the datapoints used in the computation; fairness_metric is a string denoting
    the fairness objective used; quality_fairness_notmoving_fixedeval is a dictionary of the cost of clusterings, with keys as ordering
    indices and values as the tuple (cost, fairness), where the cost was computed with respect to the original centroids found by a 
    vanilla clustering algorithm; quality_fairness_notmoving_movingeval is a dictionary of the cost of clusterings, with keys as ordering
    indices and values as the tuple (cost, fairness), where the cost was computed with respect to the actual centroids of the current clustering.
    Output: zz_notmoving_fixedeval is the list of tuples (cost, fairness) for all clusterings, where the cost was computed with respect to 
    the original centroids found by a vanilla clustering algorithm; zz_notmoving_movingeval is the list of tuples (cost, fairness) for all 
    clusterings, where the cost was computed with respect to the actual centroids of the current clustering; zz_new_notmoving_fixedeval 
    is the list of tuples (cost, fairness) for UNDOMINATED clusterings, where the cost was computed with respect to 
    the original centroids found by a vanilla clustering algorithm; zz_new_notmoving_movingeval is the list of tuples (cost, fairness) for 
    UNDOMINATED clusterings, where the cost was computed with respect to the actual centroids of the current clustering.'''
    zz_notmoving_fixedeval = {}
    zz_notmoving_movingeval = {}

    for ord in range(number_of_orderings):

        # sort by lowest cost first, highest fairness second 
        if fairness_metric == 'balance':
            zz_notmoving_fixedeval[ord] = sorted(quality_fairness_notmoving_fixedeval[ord], key=lambda x:(x[0],-x[1]))
            zz_notmoving_movingeval[ord] = sorted(quality_fairness_notmoving_movingeval[ord], key=lambda x:(x[0],-x[1]))

        elif fairness_metric == 'sumofimbalances':
            zz_notmoving_fixedeval[ord] = sorted(quality_fairness_notmoving_fixedeval[ord], key=lambda x:(x[0],x[1]))
            zz_notmoving_movingeval[ord] = sorted(quality_fairness_notmoving_movingeval[ord], key=lambda x:(x[0],x[1]))

        elif fairness_metric == 'group-util': 
            zz_notmoving_fixedeval[ord] = sorted(quality_fairness_notmoving_fixedeval[ord], key=lambda x:(x[0],x[1]))
            zz_notmoving_movingeval[ord] = sorted(quality_fairness_notmoving_movingeval[ord], key=lambda x:(x[0],x[1]))

        elif fairness_metric == 'group-util-sum':
            zz_notmoving_fixedeval[ord] = sorted(quality_fairness_notmoving_fixedeval[ord], key=lambda x:(x[0],x[1]))
            zz_notmoving_movingeval[ord] = sorted(quality_fairness_notmoving_movingeval[ord], key=lambda x:(x[0],x[1]))

        elif fairness_metric == 'group-egalit': 
            zz_notmoving_fixedeval[ord] = sorted(quality_fairness_notmoving_fixedeval[ord], key=lambda x:(x[0],x[1]))
            zz_notmoving_movingeval[ord] = sorted(quality_fairness_notmoving_movingeval[ord], key=lambda x:(x[0],x[1]))

        elif fairness_metric == 'group-egalit-sum': 
            zz_notmoving_fixedeval[ord] = sorted(quality_fairness_notmoving_fixedeval[ord], key=lambda x:(x[0],x[1]))
            zz_notmoving_movingeval[ord] = sorted(quality_fairness_notmoving_movingeval[ord], key=lambda x:(x[0],x[1]))


    zz_new_notmoving_fixedeval = {}
    for ord in range(number_of_orderings):
        zz_new_notmoving_fixedeval[ord] = []

    if fairness_metric == 'balance':
        for ord in range(number_of_orderings):

            zz_new_notmoving_fixedeval[ord].append(zz_notmoving_fixedeval[ord][0])
            init_fairness = zz_notmoving_fixedeval[ord][0][1]
            for i in zz_notmoving_fixedeval[ord]:
                # if i[1] < init_fairness: # for proportional violation objectives of sum of imbalances, which decreases 
                if i[1] > init_fairness: # for balance which should increase
                    zz_new_notmoving_fixedeval[ord].append(i)
                    init_fairness = i[1]

        zz_new_notmoving_movingeval = {}
        for ord in range(number_of_orderings):
            zz_new_notmoving_movingeval[ord] = []

        for ord in range(number_of_orderings):

            zz_new_notmoving_movingeval[ord].append(zz_notmoving_movingeval[ord][0])
            init_fairness = zz_notmoving_movingeval[ord][0][1]
            for i in zz_notmoving_movingeval[ord]:
                # if i[1] < init_fairness: # for proportional violation objectives of sum of imbalances, which decreases 
                if i[1] > init_fairness: # for balance which should increase
                    zz_new_notmoving_movingeval[ord].append(i)
                    init_fairness = i[1]
    else:
        for ord in range(number_of_orderings):

            zz_new_notmoving_fixedeval[ord].append(zz_notmoving_fixedeval[ord][0])
            init_fairness = zz_notmoving_fixedeval[ord][0][1]
            for i in zz_notmoving_fixedeval[ord]:
                if i[1] < init_fairness: # for proportional violation objectives of sum of imbalances, which decreases 
                # if i[1] > init_fairness: # for balance which should increase
                    zz_new_notmoving_fixedeval[ord].append(i)
                    init_fairness = i[1]

        zz_new_notmoving_movingeval = {}
        for ord in range(number_of_orderings):
            zz_new_notmoving_movingeval[ord] = []

        for ord in range(number_of_orderings):

            zz_new_notmoving_movingeval[ord].append(zz_notmoving_movingeval[ord][0])
            init_fairness = zz_notmoving_movingeval[ord][0][1]
            for i in zz_notmoving_movingeval[ord]:
                if i[1] < init_fairness: # for proportional violation objectives of sum of imbalances, which decreases 
                # if i[1] > init_fairness: # for balance which should increase
                    zz_new_notmoving_movingeval[ord].append(i)
                    init_fairness = i[1]

    return zz_notmoving_fixedeval, zz_notmoving_movingeval, zz_new_notmoving_fixedeval, zz_new_notmoving_movingeval

