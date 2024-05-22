''' This file contains the code for plotting Figure 2 in the main text, taking as input the output of 
Algorithm 1 (pareto_curve_alg1.py) and Algorithm 2 (repeated_FCBC_grouputil_alg2.py or repeated_FCBC_groupegalit_alg2.py).
See requirements.txt for the needed requirements to run the code.'''
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import pickle 
import json 

sns.set_style('darkgrid')

sns.set_context("paper", font_scale = 2)

if __name__ == "__main__":

    # CHANGE AS NEEDED:
    # ----------------
    max_points = 1000
    k = 2

    # fairness_metric = 'group-util' 
    fairness_metric = 'group-egalit' 

    # dataset = 'bluebike_'
    # dataset = 'census1990_'
    dataset = 'adult_'
    # ----------------

    # Read the tuples of (cost, fairness) from the pareto front file 
    file_to_read = 'output/' + dataset + 'top' + str(max_points) + 'datapoints_' + fairness_metric + '_numclusters' + str(k) + '_paretofrontmovingeval.pickle' 
    with open(file_to_read, 'rb')  as f:
        zz_new_notmoving_movingeval_optmem = pickle.load(f)

    if fairness_metric == 'group-util':

        # Reading dictionary back from JSON
        with open('output/' + dataset + 'clustering_cost' + str(max_points) + '_numclusters'+ str(k) + '_grouputil.json', 'r') as f:
            clustering_cost_loaded_data = json.load(f)
            # Convert string keys back to float
            clustering_cost_loaded_data = {float(k): v for k, v in clustering_cost_loaded_data.items()}


        # Reading dictionary back from JSON
        with open('output/' + dataset + 'group_utilitarian_obj' + str(max_points) + '_numclusters'+ str(k) + '.json', 'r') as f:
            group_obj_loaded_data = json.load(f)
            # Convert string keys back to float
            group_obj_loaded_data = {float(k): v for k, v in group_obj_loaded_data.items()}

    elif fairness_metric == 'group-egalit':


        # Reading dictionary back from JSON
        with open('output/' + dataset + 'clustering_cost' + str(max_points) + '_numclusters'+ str(k) + '_groupegalit.json', 'r') as f:
            clustering_cost_loaded_data = json.load(f)
            # Convert string keys back to float
            clustering_cost_loaded_data = {float(k): v for k, v in clustering_cost_loaded_data.items()}


        # Reading dictionary back from JSON
        with open('output/' + dataset + 'group_egalit_obj' + str(max_points) + '_numclusters'+ str(k) + '.json', 'r') as f:
            group_egalit_obj_loaded_data = json.load(f)
            # Convert string keys back to float
            group_egalit_obj_loaded_data = {float(k): v for k, v in group_egalit_obj_loaded_data.items()}

        group_obj_loaded_data = {} 
        for key in group_egalit_obj_loaded_data.keys():
            group_obj_loaded_data[key] = max(group_egalit_obj_loaded_data[key])

        
    plt.scatter(*zip(*zz_new_notmoving_movingeval_optmem[0]),s=50,alpha=0.7, label='Dyn Progr')

    plt.plot(*zip(*zz_new_notmoving_movingeval_optmem[0]), linewidth=2) 

    plt.scatter(list(clustering_cost_loaded_data.values()), list(group_obj_loaded_data.values()), s = 50, alpha=0.7, label='FCBC')

    plt.xlabel('Clustering cost')
    plt.ylabel('Proportional violation')
    plt.legend()

    if dataset == 'census1990_':
        plt.title('Census')
    elif dataset == 'adult_':
        plt.title('Adult')
    elif dataset == 'bluebike_':
        plt.title('BlueBike')
        
    plt.savefig('figures/' + dataset + str(max_points) + fairness_metric + '_numclusters' + str(k) + '_alg1alg2paretofronts.pdf',bbox_inches = 'tight')
