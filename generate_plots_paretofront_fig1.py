''' This file contains the code for plotting Figure 1 in the main text, taking as input the output of Algorithm 1 (pareto_curve_alg1.py).
See requirements.txt for the needed requirements to run the code.'''
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import pickle 

sns.set_style('darkgrid')

sns.set_context("paper", font_scale = 2)

if __name__ == "__main__":

    # CHANGE AS NEEDED:
    # ----------------
    max_points = 1000
    k = 2

    # fairness_metric = 'balance'
    fairness_metric = 'group-util' 
    # fairness_metric = 'group-util-sum' 
    # fairness_metric = 'group-egalit' 
    # fairness_metric = 'group-egalit-sum'
    # fairness_metric = 'sumofimbalances'

    dataset = 'bluebike_'
    # dataset = 'census1990_'
    # dataset = 'adult_'
    # ----------------

    # Read the tuples of (cost, fairness) from the pareto front file 
    file_to_read = 'output/' + dataset + 'top' + str(max_points) + 'datapoints_' + fairness_metric + '_numclusters' + str(k) + '_paretofrontmovingeval.pickle' 
    with open(file_to_read, 'rb')  as f:
        zz_new_notmoving_movingeval_optmem = pickle.load(f)

    plt.scatter(*zip(*zz_new_notmoving_movingeval_optmem[0]),s=50,alpha=0.7) 

    plt.plot(*zip(*zz_new_notmoving_movingeval_optmem[0]), linewidth=2) 

    plt.xlabel('Clustering cost')
    if fairness_metric == 'balance':
        plt.ylabel('Balance')
    elif fairness_metric == 'sumofimbalances':
        plt.ylabel('Sum of Imbalances')
    else:
        plt.ylabel('Proportional violation')

    if dataset == 'census1990_':
        plt.title('Census')
    elif dataset == 'adult_':
        plt.title('Adult')
    elif dataset == 'bluebike_':
        plt.title('BlueBike')
        
    plt.savefig('figures/' + dataset + str(max_points) + 'datapoints_' + fairness_metric + '_numclusters' + str(k) + '_paretofront.pdf',bbox_inches = 'tight')