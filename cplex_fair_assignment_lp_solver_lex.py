''' This code is adapted from https://github.com/Seyed2357/Fair-Clustering-Under-Bounded-Cost and used for the repeated FCBC algorithm (Algorithm 2).'''
import numpy as np
from scipy.spatial.distance import cdist
from cplex import Cplex
import time
from nf_rounding_multi_color import min_cost_rounding_multi_color 
import timeit
from util.pof_util_lex import relax_alpha_viol, relax_beta_viol, get_color_with_min_proportion, relax_util_viol, alpha_beta_array, get_lex_multi_color, get_util_value_two_color, relax_lex_viol, relax_lex_viol_possible, relax_lex_viol_all_fixed, H_fixed_one_set
import copy 


tolerance = 0

# L_POF is the lower bound on the cluster size for rounding 

def fair_partial_assignment_lex(df, centers, initial_score, delta, color_proprtions, alpha, beta, color_flag, clustering_method, num_colors, L=0, epsilon=1/8, alpha_POF=10,L_POF=-1):


    round_POF = 2/L_POF

    if clustering_method == "kmeans" or clustering_method == "kmedian":
        cost_fun_string = 'euclidean' if clustering_method == "kmedian" else 'sqeuclidean'


        # r_min (r_max) is the smallest (largets) proportion, col_min (col_max) is the least (most) color 
        r_min, col_min, r_max, col_max = get_color_with_min_proportion(color_proprtions) 
        beta_min = r_min*(1-delta)
        beta_max = r_max*(1-delta)
        alpha_min = r_min*(1+delta)
        alpha_max = r_max*(1+delta)

        # num_colors is the number of colors 
        num_colors = len(color_proprtions)

        # delta_lex is the set of violations in the leximin solution 
        delta_lex = {key: -1 for key in list(range(num_colors))}

        # H_fixed is the set of colors no longer improvable 
        H_fixed = {key: set() for key in list(range(num_colors))}

        # H_all is the set of all colors 
        H_all = set(range(num_colors))

        # H_movable 
        H_movable = H_all.difference(H_fixed_one_set(H_fixed))



        alpha_orig = copy.deepcopy(alpha)
        beta_orig = copy.deepcopy(beta)

        
        E_eps = np.linspace(start=0, stop=1-epsilon, num=int(1/epsilon), endpoint=True)


        lex_counter = 0 
        counter = 0 



        while H_fixed_one_set(H_fixed) != H_all:


            print('counter = %d' %counter)
            print('lex_counter = %d' %lex_counter)
            print('H_all')
            print(H_all)
            print('H_fixed')
            print(H_fixed)



            ## This is where the oteration begins 

            # initialize for the iteration 
            L_viol_idx = 0
            R_viol_idx = E_eps.shape[0] - 1
            best_viol = -1 


            while (L_viol_idx <= R_viol_idx):
                counter += 1 
                m_viol_idx = int(np.floor((L_viol_idx + R_viol_idx) / 2)) 
                alpha_relax, beta_relax = relax_lex_viol(alpha_orig,beta_orig,H_fixed,delta_lex,E_eps[m_viol_idx])
 

                problem, objective = fair_partial_assignment_lp_solver(df, centers, initial_score, alpha_POF, color_flag, alpha_relax, beta_relax, cost_fun_string, L, clustering_method)
                problem.solve()

                # if the problem is infeasible, then you need a higher violation 
                if (problem.solution.get_status() == 3):
                    L_viol_idx = m_viol_idx + 1

                elif (problem.solution.get_status() == 1):
                    R_viol_idx = m_viol_idx - 1
                    best_viol = m_viol_idx 



            # once the minimum violation is found, it has to be pushed up because of the rounding 
            eps_best_viol = min( E_eps[best_viol] + round_POF , 1-tolerance) 
            delta_lex[lex_counter] = eps_best_viol

            none_fixed = False

            for h_color in H_movable:
                ''' 
                print('check relax_lex_viol_possible')
                print('h_color')
                print(h_color)

                print('H_fixed')
                print(H_fixed)

                print('H_movable')
                print(H_movable)

                print('alpha_orig')
                print(alpha_orig)

                print('beta_orig')
                print(beta_orig)

                print('eps_best_viol')
                print(eps_best_viol)

                print('eps_best_vio-round_POF-epsilon')
                print(eps_best_viol-round_POF-epsilon)
                ''' 

                alpha_relax, beta_relax = relax_lex_viol_possible(H_all, H_fixed, delta_lex, alpha_orig, beta_orig, eps_best_viol, h_color, round_POF, epsilon)
                
                #print('alpha_relax')
                #print(alpha_relax)
                #print('beta_relax')
                #print(beta_relax)

                problem, objective = fair_partial_assignment_lp_solver(df, centers, initial_score, alpha_POF, color_flag, alpha_relax, beta_relax, cost_fun_string, L, clustering_method)
                problem.solve()

                # if possible to move h_color 
                if (problem.solution.get_status() == 1):
                    # if have reached the minimum possible 
                    if (eps_best_viol-round_POF-epsilon) <= E_eps[1]:
                        H_fixed[lex_counter] = H_fixed[lex_counter].union( set([h_color]))
                        none_fixed = True 
                # if NOT possible to move h_color 
                # POF 
                elif (problem.solution.get_status() == 3):
                    H_fixed[lex_counter] = H_fixed[lex_counter].union( set([h_color]))
                    none_fixed = True 


            # if each can move individually \
            ''' 
            if none_fixed==False:
                print('if each can move individually')
                print('H_fixed before')
                print(H_fixed)
                for h_color in H_movable:
                    H_fixed[lex_counter] = H_fixed[lex_counter].union( set([h_color]))


                print('H_fixed after')
                print(H_fixed)
                print(type(H_fixed)) 
            ''' 

            # if each can move individually \
            # this picks one color randomly to be fixed in case all can improve 
            if none_fixed==False:
                numCurrentCols = len(H_movable)
                randomColIdx = np.random.random_integers(0,numCurrentCols-1)
                for i, h_mov in enumerate(H_movable):
                    if i==randomColIdx:
                        H_fixed[lex_counter] = H_fixed[lex_counter].union( set([h_mov]))

            #    for h_color in H_movable:
            #        H_fixed[lex_counter] = H_fixed[lex_counter].union( set([h_color]))


            #print('updated H_fixed')
            #print(H_fixed)


            #print('H_movable')
            #print(H_movable)

            print('H_fixed changed')
            print(H_fixed)

            print('H_movable not changed')
            print(H_movable)
            print(type(H_movable))
            print(len(H_movable))

            H_movable = H_all.difference(H_fixed_one_set(H_fixed))

            print('H_movable changed')
            print(H_movable)


            #print('updated H_movable')
            #print(H_movable)

            lex_counter += 1 








        # Step 5. call the solver
        util_lp = -1


        alpha_relax, beta_relax = relax_lex_viol_all_fixed(alpha_orig, beta_orig, H_fixed, delta_lex)
        problem, objective = fair_partial_assignment_lp_solver(df, centers, initial_score, alpha_POF, color_flag, alpha_relax, beta_relax, cost_fun_string, L, clustering_method)
        problem.solve()



        t1 = time.monotonic()
        problem.solve()
        t2 = time.monotonic()
        print("LP solving time = {}".format(t2-t1))

        # problem.solution is a weakly referenced object, so we must save its data
        #   in a dictionary so we can write it to a file later.
        res = {
            "partial_status": problem.solution.get_status(),
            "partial_success": problem.solution.get_status_string(),
            "partial_objective": problem.solution.get_objective_value(),
            "partial_assignment": problem.solution.get_values(),
        }

        # NOTE: here is where the rounding is done 
        # comment the first line and uncomment the second to disable the rounnding
        #final_res = iterative_rounding_lp(df, centers, objective, color_flag, res)
        start_time_nf = timeit.default_timer()
        final_res = min_cost_rounding_multi_color(df, centers, objective, color_flag, num_colors, res) 
        elapsed_time_nf = timeit.default_timer() - start_time_nf


        final_res["partial_assignment"] = res["partial_assignment"]
        final_res["partial_objective"] = res["partial_objective"]
        final_res["bs_iterations"] = counter
        final_res["opt_index"] = best_viol
        final_res['util_lp'] = util_lp

        if clustering_method == "kmeans":
            final_res["partial_objective"] = np.sqrt(final_res["partial_objective"])
            final_res["objective"] = np.sqrt(final_res["objective"])

        return final_res, elapsed_time_nf, delta_lex, H_fixed

    elif clustering_method == "kcenter":
        problem, objective = fair_partial_assignment_lp_solver(df, centers, initial_score, color_flag, alpha, beta, 'sqeuclidean',L,clustering_method)
        problem.solve()

        cost_ub = max(objective) + 1
        cost_lb = 0
        lowest_feasible_cost = cost_ub
        cheapest_feasible_lp = problem
        cheapest_feasible_obj = objective

        while cost_ub > cost_lb + 0.1:
            cost_mid = (cost_ub + cost_lb)/2.0
            new_problem, new_objective = fair_partial_assignment_lp_solver(df, centers, initial_score, color_flag, alpha, beta, 'sqeuclidean', L,clustering_method)
            to_delete = [idx for idx, el in enumerate(objective) if el > cost_mid]
            # print("to delete : {}".format(to_delete))
            if len(to_delete) > 0:
                new_problem.variables.delete(to_delete)

            new_problem.solve()
            new_stats = new_problem.solution.get_status()
            if new_stats == 1: # optimal!
                cost_ub = cost_mid
                lowest_feasible_cost = cost_mid
                cheapest_feasible_lp = new_problem
                cheapest_feasible_obj = new_objective

            elif new_stats == 3: #infeasible
                cost_lb = cost_mid

            else:
                raise ValueError("LP solver stat code {}".format(new_stats) + " with cost {}".format(cost_mid))

        # Make "objective" and "assignment" arrays that match the original number of variables
        num_centers = len(centers)
        nr_variables = len(objective)
        assignment = [0]*nr_variables
        objective = [0]*nr_variables

        for new_idx, var_name in enumerate(cheapest_feasible_lp.variables.get_names()):
            parts = var_name.split('_')
            j = int(parts[1]) # point number
            i = int(parts[2]) # center number
            old_idx = j*num_centers + i
            # old_idx = problem.variables.get_index(var_name)

            old_name = problem.variables.get_names(old_idx)
            if old_name != var_name:
                raise Exception("Old name: {} and var_name: {} do not match for new_idx = {} and old_idx = {}".format(old_name, var_name, new_idx, old_idx))
            objective[old_idx] = cheapest_feasible_obj[new_idx]
            assignment[old_idx] = cheapest_feasible_lp.solution.get_values(new_idx)


        # problem.solution is a weakly referenced object, so we must save its data
        #   in a dictionary so we can write it to a file later.
        res = {
            "status" : cheapest_feasible_lp.solution.get_status(),
            "success" : cheapest_feasible_lp.solution.get_status_string(),
            "objective" : cheapest_feasible_lp.solution.get_objective_value(),
            "assignment" : assignment,
        }

        #final_res = iterative_rounding_lp(df, centers, objective, color_flag, res)
        final_res = min_cost_rounding_multi_color(df, centers, objective, color_flag, num_colors, res) 
        
        final_res["partial_assignment"] = res["assignment"]
        final_res["partial_objective"] = res["objective"]
        
        # check sol value 
        lp_sol_val = final_res["partial_objective"]
        rounded_sol_val = final_res["objective"]

        rounded_cost = 0
        for idx, value in enumerate(final_res["assignment"]):
            rounded_cost = max(rounded_cost, value * objective[idx])

        final_res["objective"] = np.sqrt(rounded_cost)
        final_res["partial_objective"] = np.sqrt(lowest_feasible_cost)
        final_res["partial_assignment"] = assignment

        return final_res

    else:
        print("Not a valid clustering method. Available methods are: " 
              "\'kmeans\', \'kmedian\', and \'kcenter\'.")
        return None


'''
The main function in this file is fair_partial_assignment_lp_solver.
This function takes as input a collection of data points, a list of 
cluster centers, a list of colors of each points, and fairness parameters.
It then constructs the fair assignment lp and solves it. It returns 
a fractional assignment of each point to a cluster center.  

Input Description:
    df: a dataframe of the input points
    centers: a list of the euclidean centers found via clustering
    color_flag : a list of color values for all the points -- helpful in adding constraints to the lp solver
    alpha: dict where the keys are colors and the values is the alpha for that color
    beta: dict where the keys are colors and the values are the beta for that color

Output Description:
    res: a dictionary with the following keys:
        "status": an integer code depicting the outcome of the lp solver
        "success": a string to interpret the above code
        "objective": the objective function value
        "assignment": the assignment of values to all the LP variables
        
Reading the "assignment" array:
    the variables are created in the following order:
        for all j in points
            for all i in centers 
                create x_{j}_{i} 
    So the assignment array, which is a list of floats, corresponds to this variable order
'''


def fair_partial_assignment_lp_solver(df, centers, initial_score, alpha_POF, color_flag, alpha, beta, cost_fun_string, L,clustering_method):

    # There are primarily five steps:
    # 1. Initiate a model for cplex
    # 2. Declare if it is minimization or maximization problem
    # 3. Add variables to the model. The variables are generally named.
    #    The upper bounds and lower bounds on the range for the variables
    #    are also mentioned at this stage. The coefficient of the objective
    #    functions are also entered at this step
    # 4. Add the constraints to the model. The constraint matrix, denoted by A,
    #    can be added in three ways - row wise, column wise or non-zero entry wise.
    # 5. Finally, call the solver.

    # Step 1. Initiate a model for cplex.

    print("Initializing Cplex model")
    problem = Cplex()

    # Step 2. Declare that this is a minimization problem

    problem.objective.set_sense(problem.objective.sense.minimize)

    # Step 3.   Declare and  add variables to the model. The function
    #           prepare_to_add_variables (points, center) prepares all the
    #           required information for this stage.
    #
    #    objective: a list of coefficients (float) in the linear objective function
    #    lower bounds: a list of floats containing the lower bounds for each variable
    #    upper bounds: a list of floats containing the upper bounds for each variable
    #    variable_name: a list of strings that contains the name of the variables

    print("Starting to add variables...")
    t1 = time.monotonic()

    # NOTE: lower_bounds=0, upper_bounds = 1, variable_names = {x_j_i}, objective=list of corresponding coefficients for each x_i_j 
    objective, lower_bounds, upper_bounds, variable_names = prepare_to_add_variables(df, centers, cost_fun_string)

    #print('investigate')
    #print(objective)
    #print(type(objective))
    #print(len(objective))



    problem.variables.add(obj=objective,
                          lb=lower_bounds,
                          ub=upper_bounds,
                          names=variable_names)
    t2 = time.monotonic()
    print("Completed. Time for creating and adding variable = {}".format(t2-t1))

    # Step 4.   Declare and add constraints to the model.
    #           There are few ways of adding constraints: rwo wise, col wise and non-zero entry wise.
    #           Assume the constraint matrix is A. We add the constraints row wise.
    #           The function prepare_to_add_constraints_by_entry(points,center,colors,alpha,beta)
    #           prepares the required data for this step.
    #
    #  constraints_row: Encoding of each row of the constraint matrix
    #  senses: a list of strings that identifies whether the corresponding constraint is
    #          an equality or inequality. "E" : equals to (=), "L" : less than (<=), "G" : greater than equals (>=)
    #  rhs: a list of floats corresponding to the rhs of the constraints.
    #  constraint_names: a list of string corresponding to the name of the constraint

    print("Starting to add constraints...")
    t1 = time.monotonic()
    objects_returned = prepare_to_add_constraints(df, centers, initial_score, alpha_POF, color_flag, beta, alpha, L, clustering_method)
    constraints_row, senses, rhs, constraint_names = objects_returned
    problem.linear_constraints.add(lin_expr=constraints_row,
                                   senses=senses,
                                   rhs=rhs,
                                   names=constraint_names)
    t2 = time.monotonic()
    print("Completed. Time for creating and adding constraints = {}".format(t2-t1))

    # Optional: We can set various parameters to optimize the performance of the lp solver
    # As an example, the following sets barrier method as the lp solving method
    # The other available methods are: auto, primal, dual, sifting, concurrent

    #problem.parameters.lpmethod.set(problem.parameters.lpmethod.values.barrier)







    return problem, objective

# This function creates necessary stuff to prepare for variable
# addition to the cplex lp solver
# Input: See the input description of the main function
# Output:
#   objective: a list of coefficients (float) in the linear objective function
#   lower bounds: a list of floats containing the lower bounds for each variable
#   upper bounds: a list of floats containing the upper bounds for each variable
#   variable_name: a list of strings that contains the name of the variables
# The ordering of variables are as follows: for every point, we create #centers many
# variables and put them together before moving onto the next point.
# for j in range(num_points)
#   for i in range(num_centers)
#       create x_j_i


def prepare_to_add_variables(df, centers, cost_fun_string):

    num_points = len(df)
    num_centers = len(centers)

    # NOTE: this reverses the order we are used to x_i_j, this is x_j_i 
    # Name the variables -- x_j_i is set to 1 if j th pt is assigned to ith center
    variable_names = ["x_{}_{}".format(j,i) for j in range(num_points) for i in range(num_centers)]

    # All values should be between 0 and 1 -- if a single tuple is provided,
    #   then it will be applied to all points, according to scipy docs.
    total_variables = num_points * num_centers
    
    # NOTE: below is always true 
    assert len(variable_names) == total_variables


    lower_bounds = [0 for _ in range(total_variables)]
    upper_bounds = [1 for _ in range(total_variables)]

    # Cost function: Minimize the weighted sum of the distance from each point
    #   to each center.
    objective = cost_function(df, centers, cost_fun_string)

    return objective, lower_bounds, upper_bounds, variable_names

# Cost function: Minimize the weighted sum of the distance from each point
#   to each center.
# Implementation details:
#   cdist(XA, XB, metric='euclidean', *args, **kwargs): Compute distance between each pair of the two
#   collections of inputs. metric = 'sqeuclidean' computes the squared Euclidean distance. cdist returns a
#   |XA| by |XB| distance matrix. For each i and j, the metric dist(u=XA[i], v=XB[j]) is computed and stored in the
#   (i,j)th entry.
#   ravel(a,order='C'): A 1-D array, containing the elements of the input, is returned. A copy is made only if needed.
# The ordering of the final output array is consistent with the order of the variables.
# Note that all_pair_distance is an array and we need to convert it to a list before returning


def cost_function(df,centers, cost_fun_string):
    ''' 
    print('WEEEEEEEEEEEEEEEEEEEEEEEEE')
    print(type(centers))
    print(type(df))
    print(centers)
    ''' 
    # NOTE: centers is a list of rows of the centers 
    # df is a data frame
    print('HERE FIRST')
    print(type(centers))
    print(centers)
    all_pair_distance = cdist(df.values,centers,cost_fun_string)
    # MOTE: ravel() will stack things row by row into one big row 
    #print('check the distances')
    #print(cost_fun_string)
    #print(all_pair_distance)
    return all_pair_distance.ravel().tolist()

# This function prepares for constraints addition by non zero entries
# Input: See the input description of the main function
# Output:
#  constraints_row: Encoding of each row of the constraint matrix
#  senses: a list of strings that identifies whether the corresponding constraint is
#          an equality or inequality. "E" : equals to (=), "L" : less than (<=), "G" : greater than equals (>=)
#  rhs: a list of floats corresponding to the rhs of the constraints.
#  constraint_names: a list of string corresponding to the name of the constraint
#
# There are two types of constraints.
# 1. Assignment constraints and 2. Fairness constraints
#     1. first we have no_of_points many assignment constraints (sum_{i} x_j_i = 1 for all j)
#     2. then we have num_center*num_colors*2 many fairness constraints.
#        fairness constraints are indexed by center, and then color. We first have all beta constraints, followed
#        by alpha constraints


def prepare_to_add_constraints(df, centers, initial_score, alpha_POF, color_flag, beta, alpha, L, clustering_method):

    num_points = len(df)
    num_centers = len(centers)


    # The following steps constructs the assignment constraint. Each corresponding row in A
    # has #centers many non-zero entries.
    constraints_row, rhs = constraint_sums_to_one(num_points, num_centers)
    sum_const_len = len(rhs)


    # The following steps constructs the fairness constraints. There are #centers * # colors * 2
    # many such constraints. Each of them has #points many non-zero entries.
    for var in color_flag:
        var_color_flag, var_beta, var_alpha = color_flag[var], beta[var], alpha[var]
        ''' NOTE:
        print('HHHEEHHEHEHEHEHEHEHEHEHEHEHEHEH')
        print(var)
        print('HHHEEHHEHEHEHEHEHEHEHEHEHEHEHEH')
        print(var_color_flag)
        print('HHHEEHHEHEHEHEHEHEHEHEHEHEHEHEH')
        print(var_beta)
        print('HHHEEHHEHEHEHEHEHEHEHEHEHEHEHEH')ç
        print(var_alpha)
        ''' 
        color_constraint, color_rhs = constraint_color(num_points, num_centers, var_color_flag, var_beta, var_alpha)
        constraints_row.extend(color_constraint)
        rhs.extend(color_rhs)


    # NOTE: This is to enforce the lower bound
    # If L=0, then no lower bound is enforced 
    constraints_LB, rhs_LB = constraint_lower_bound_cluster(num_points, num_centers, L)
    constraints_row.extend(constraints_LB)
    rhs.extend(rhs_LB)

    # POF:
    #print('check what can happen -kmeans')
    # set U_POF to the correct value according to the objective 
    power_p = 1 if clustering_method == "kmedian" else 2
    U_POF = (initial_score*alpha_POF)**(power_p) #100000 
    constraints_POF, rhs_POF = constraint_POF(df, centers, num_points, num_centers, U_POF, clustering_method) 
    constraints_row.extend(constraints_POF)
    rhs.extend(rhs_POF)
    #print(rhs_POF)
    #print('check what can happen Done POF POF POF')
    #print(constraints_POF)
    #print(rhs_POF)


    # The assignment constraints are of equality type and the rest are less than equal to type
    # NOTE: no need to add anything for lower bounded cluster size 
    senses = ["E" for _ in range(sum_const_len)] + ["L" for _ in range(len(rhs) - sum_const_len)]  

    # Name the constraints
    constraint_names = ["c_{}".format(i) for i in range(len(rhs))]

    return constraints_row, senses, rhs, constraint_names


# this function adds the constraint that every client must be assigned to exactly one center
# Implementation:
# Example: assume 3 points and 2 centers. Total of 6 variables: x_0_0,x_0_1 for first point, and so on.
#  row 0         x00 + x01                 = 1
#  row 1                  x10 + x11        = 1
#  row 2                         x20 + x21 = 1
# The constraints are entered in the following format:
# [[['x_0_0', 'x_0_1'], [1, 1]],
#  [['x_1_0', 'x_1_1'], [1, 1]],
#  [['x_2_0', 'x_2_1'], [1, 1]]]


def constraint_sums_to_one(num_points, num_centers):

    constraints = [[["x_{}_{}".format(j, i) for i in range(num_centers)], [1] * num_centers] for j in range(num_points)]
    rhs = [1] * num_points
 
    ''' NOTE: below just verifies 
    print('VERIFY')
    print(len(constraints))
    print(len(rhs))
    print(constraints[0])
    ''' 
    return constraints, rhs


# this function adds the fairness constraint
# Implementation:
# The following example demonstrates this
# Example:  Assume 3 points and 2 centers and 2 colors.
#           Total of 6 variables: x00,x01 for first point, x10,x11 for second point and so on.
#           Assume 1 and 2 belong to color 0. Let a == alpha and b == beta
#
#     (b1-1) x00             + (b1-1) x10            + b1 x20                 <= 0    center 0, color 1, beta
#        b2  x00             +   b2   x10            + (b2-1) x20             <= 0    center 0, color 2, beta
#                 (b1-1) x01           + (b1-1) x11              + b1 x21     <= 0    center 1, color 1, beta
#                    b2  x01            +  b2   x11              + (b2-1) x21 <= 0    center 1, color 2, beta
#
#     (1-a1) x00            + (1-a1) x20             - a1 x30                 <= 0    center 1, color 1, alpha
#       - a2 x00             - a2 x20                + (1-a2) x30             <= 0    center 1, color 2, alpha
#               (1-a1) x10             + (1-a1) x21              - a1 x31     <= 0    center 2, color 1, alpha
#              - a2 x10                 - a2   x21              + (1-a2) x31 <= 0    center 2, color 2, alpha
#
# Below we depict the details of the entries (the first 4 rows)
# [
# [['x_0_0','x_1_0','x_2_0'],[b1-1,b1-1,b1]]
# [['x_0_0','x_1_0','x_2_0'],[b2,b2,b2-1]]
# [['x_0_1','x_1_1','x_2_1'],[b1-1,b1-1,b1]]
# [['x_0_1','x_1_1','x_2_1'],[b2,b2,b2-1]]
# ...]

def constraint_color(num_points, num_centers, color_flag, beta, alpha):

    beta_constraints = [[["x_{}_{}".format(j, i) for j in range(num_points)],
                         [beta[color] - 1 if color_flag[j] == color else beta[color] for j in range(num_points)]]
                        for i in range(num_centers) for color, _ in beta.items()]
    alpha_constraints = [[["x_{}_{}".format(j, i) for j in range(num_points)],
                          [np.round(1 - alpha[color], decimals=3) if color_flag[j] == color else (-1) * alpha[color]
                           for j in range(num_points)]]
                         for i in range(num_centers) for color, _ in beta.items()]
    constraints = beta_constraints + alpha_constraints
    number_of_constraints = num_centers * len(beta) * 2
    rhs = [0] * number_of_constraints
    return constraints, rhs




# NOTE: This forces the minimum number of points assigned to a cluster to be >=L 
def constraint_lower_bound_cluster(num_points, num_centers, L):

    constraints = [[["x_{}_{}".format(j, i) for j in range(num_points)], [-1] *  num_points] for i in range(num_centers)]
    rhs = [-L]*num_centers
 
    print('\n LB constraints')
    print(len(constraints))

    return constraints, rhs


# POF: Constraint on the POF
def constraint_POF(df, centers, num_points, num_centers, U_POF, clustering_method):

    cost_fun_string = 'euclidean' if clustering_method == "kmedian" else 'sqeuclidean'

    variable_names = ["x_{}_{}".format(j,i) for j in range(num_points) for i in range(num_centers) ]
    distances_p = cost_function(df,centers, cost_fun_string)


    constraints = [[variable_names,distances_p]]
    rhs = [U_POF]*1


    return constraints, rhs
