''' This code is adapted from https://github.com/Seyed2357/Fair-Clustering-Under-Bounded-Cost and used for the repeated FCBC algorithm (Algorithm 2).'''
import configparser
import time
from collections import defaultdict
from functools import partial
import numpy as np
import pandas as pd
from cplex_fair_assignment_lp_solver_util import fair_partial_assignment_util
from util.clusteringutil import (clean_data, read_data, scale_data,
                                 subsample_data, take_by_key,
                                 vanilla_clustering, write_fairness_trial)
from util.configutil import read_list
from util.probutil import form_class_prob_vector, sample_colors, create_prob_vecs, sample_colors_ml_model
from util.pof_util import relax_alpha_viol, relax_beta_viol, get_color_with_min_proportion, relax_util_viol, alpha_beta_array, get_viol_value_two_color, get_util_value_two_color
from util.utilhelpers import get_clust_sizes, max_Viol_multi_color, x_for_colorBlind, max_RatioViol_multi_color, find_proprtions_two_color_deter 
import copy



# This function takes a dataset and performs a fair clustering on it.
# Arguments:
#   dataset (str) : dataset to use
#   config_file (str) : config file to use (will be read by ConfigParser)
#   data_dir (str) : path to write output
#   num_clusters (int) : number of clusters to use
#   deltas (list[float]) : delta to use to tune alpha, beta for each color
#   max_points (int ; default = 0) : if the number of points in the dataset 
#       exceeds this number, the dataset will be subsampled to this amount.
# Output:
#   None (Writes to file in `data_dir`)  
def fair_clustering_util(counter, initial_score_save, pred_save, cluster_centers_save,dataset, config_file, data_dir, num_clusters, deltas, max_points, L=0, p_acc=1.0, ml_model_flag=False,two_color_util=True,epsilon=0.0,alpha_POF=0):
    config = configparser.ConfigParser(converters={'list': read_list})
    config.read(config_file)

    # Read data in from a given csv_file found in config
    # df (pd.DataFrame) : holds the data
    df = read_data(config, dataset)

    # Subsample data if needed
    if max_points and len(df) > max_points:
       df = df.head(max_points)


    #print(df.head(10))


    # Clean the data (bucketize text data)
    df, _ = clean_data(df, config, dataset)


    # variable_of_interest (list[str]) : variables that we would like to collect statistics for
    variable_of_interest = config[dataset].getlist("fairness_variable")
    
    # NOTE: this code only handles one color per vertex 
    assert len(variable_of_interest) == 1 

    # Assign each data point to a color, based on config file
    # attributes (dict[str -> defaultdict[int -> list[int]]]) : holds indices of points for each color class
    # color_flag (dict[str -> list[int]]) : holds map from point to color class it belongs to (reverse of `attributes`)
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

    # Select only the desired columns
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


    t1 = time.monotonic()

    if counter ==0:
        initial_score, pred, cluster_centers = vanilla_clustering(df, num_clusters, clustering_method)
        initial_score_save, pred_save, cluster_centers_save = initial_score, pred, cluster_centers
    else:
        initial_score, pred, cluster_centers = initial_score_save, pred_save, cluster_centers_save

    t2 = time.monotonic()
    cluster_time = t2 - t1
    print("Clustering time: {}".format(cluster_time))
    


    # sizes (list[int]) : sizes of clusters
    sizes = [0 for _ in range(num_clusters)]
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



        # Solves partial assignment and then performs rounding to get integral assignment
        t1 = time.monotonic()
        res, nf_time, r_min, col_min, r_max, col_max = fair_partial_assignment_util(df, cluster_centers, initial_score, delta, color_proprtions, fp_alpha, fp_beta, fp_color_flag, clustering_method, num_colors, L, epsilon,alpha_POF)
        t2 = time.monotonic()
        lp_time = t2 - t1



        ### Output / Writing data to a file
        # output is a dictionary which will hold the data to be written to the
        #   outfile as key-value pairs. Outfile will be written in JSON format.
        output = {}

        # num_clusters for re-running trial
        output["num_clusters"] = num_clusters

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
        output["cluster_time"] = cluster_time

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

        # Record Probability Vector
        # NOTE: TODO  

        # Writes the data in `output` to a file in data_dir
        write_fairness_trial(output, data_dir)

        # Added because sometimes the LP for the next iteration solves so 
        # fast that `write_fairness_trial` cannot write to disk
        time.sleep(1) 

        viol_upper,viol_lower = get_viol_value_two_color(np.reshape(output['proportions_normalized'], (2,-1)),alpha_orig,beta_orig)
        util_objective = get_util_value_two_color(np.reshape(output['proportions_normalized'], (2,-1)),alpha_orig,beta_orig)

        ( _ , color_flag), = color_flag.items()



        rounded_prop, _ , dummy = find_proprtions_two_color_deter(np.reshape(res["assignment"], (-1,num_clusters)) ,2,color_flag,num_clusters)
        lp_prop, _ , dummy = find_proprtions_two_color_deter(np.reshape(res["partial_assignment"], (-1,num_clusters)) ,2,color_flag,num_clusters)




        output['util_objective'] = util_objective 
        output["bs_iterations"] = res['bs_iterations']
        output["epsilon"] = epsilon
        output["epsilon set size "] = 1/epsilon
        output["alpha_pof"] = alpha_POF
        output['upper_violations'] = viol_upper.ravel().tolist()
        output['lower_violations'] = viol_lower.ravel().tolist()
        output['opt_index'] = res['opt_index']
        output['util_lp']  = res['util_lp'] 
        output['color_flag'] = color_flag



        return output, initial_score_save, pred_save, cluster_centers_save
