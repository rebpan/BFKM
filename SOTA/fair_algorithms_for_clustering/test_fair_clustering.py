import configparser
from util.configutil import read_list
import sys
import numpy as np
from sklearn.preprocessing import scale
from cplex import Cplex
from scipy.spatial.distance import pdist,squareform


def fair_clustering(dataset, data_dir, n_clusters, delta, max_points, violating, violation):
    X_org = np.loadtxt(r'../../Datasets/subsampled_bank.txt', delimiter=',')
    Color = np.loadtxt(r'../../Datasets/subsampled_bank_Color.txt', dtype=int)
    # Processing Color
    uq_color = np.unique(Color)
    attribute = {uq_color[k] : list(np.where(Color == uq_color[k])[0]) for k in range(len(uq_color))}
    representation = {uq_color[k] : len(attribute[k])/len(Color) for k in range(len(uq_color))}
    # standardization and normalization
    X = scale(X_org, axis = 0)
    feanorm = np.maximum(1e-14,np.sum(X**2,axis=1))
    X = X/(feanorm[:,None]**0.5)

    if not violating:
        pass
    else:
        cluster_time, initial_score = 0, 0
        fairness, ratio = {}, {}
        sizes, cluster_centers = [], []

    # bound representation
    alpha, beta = {}, {}
    a_val, b_val = 1 / (1 - delta), 1 - delta
    alpha = {uq_color[k] : a_val * representation[k] for k in range(len(uq_color))}
    beta = {uq_color[k] : b_val * representation[k] for k in range(len(uq_color))}

    if not violating:
        pass
    else:
        res = violating_lp_clustering(X, n_clusters, alpha, beta, Color, violation)
        res["partial_objective"] = 0
        res["partial_assignment"] = []

    output = {}
    # Whether or not the LP found a solution
    output["success"] = res["success"]
    # Nonzero status -> error occurred
    output["status"] = res["status"]
    # Save alphas and betas from trials
    output["alpha"] = alpha
    output["beta"] = beta
    # Save original clustering score
    #output["unfair_score"] = initial_score
    # Clustering score after addition of fairness
    output["fair_score"] = res["objective"]
    output["assignment"] = res["assignment"]
    output["partial_assignment"] = res["partial_assignment"]
    print(output["assignment"])


def violating_lp_clustering(X, n_clusters, alpha, beta, Color, violation):
    problem, objective = violating_clustering_lp_solver(X, n_clusters, Color, alpha, beta, violation)

    problem.solve()
    print("LP solved")

    objective_value = problem.solution.get_objective_value()
    objective_value = np.sqrt(objective_value)
    res = {
        "status": problem.solution.get_status(),
        "success": problem.solution.get_status_string(),
        "objective": objective_value,
        "assignment": problem.solution.get_values(),
    }
    return res


def violating_clustering_lp_solver(X, n_clusters, Color, alpha, beta, violation):

    # Step 1. Initiate a model for cplex.
    problem = Cplex()
    # Step 2. Declare that this is a minimization problem
    problem.objective.set_sense(problem.objective.sense.minimize)
    # Step 3.   Declare and  add variables to the model. The function
    #           prepare_to_add_variables prepares all the
    #           required information for this stage.
    #
    #    objective: a list of coefficients (float) in the linear objective function
    #    lower bounds: a list of floats containing the lower bounds for each variable
    #    upper bounds: a list of floats containing the upper bounds for each variable
    #    variable_names: a list of strings that contains the name of the variables
    print("Starting to add variables...")
    objective, lower_bounds, upper_bounds, variable_names = prepare_to_add_variables(X)
    problem.variables.add(obj=objective, lb=lower_bounds, ub=upper_bounds, names=variable_names)
    print("Completed")
    # Step 4.   Declare and add constraints to the model.
    #           There are few ways of adding constraints: rwo wise, col wise and non-zero entry wise.
    #           The function prepare_to_add_constraints_by_entry
    #           prepares the required data for this step. Assume the constraint matrix is A.
    #  constraints_row: Encoding of each row of the constraint matrix
    #  senses: a list of strings that identifies whether the corresponding constraint is
    #          an equality or inequality. "E" : equals to (=), "L" : less than (<=), "G" : greater than equals (>=)
    #  rhs: a list of floats corresponding to the rhs of the constraints.
    #  constraint_names: a list of string corresponding to the name of the constraint
    print("Starting to add constraints...")
    objects_returned = prepare_to_add_constraints(X, n_clusters, Color, beta, alpha, violation)
    print(objects_returned)
    constraints_row, senses, rhs, constraint_names = objects_returned

    problem.linear_constraints.add(lin_expr=constraints_row, senses=senses, rhs=rhs, names=constraint_names)
    print("Completed")

    # Optional
    problem.parameters.lpmethod.set(problem.parameters.lpmethod.values.barrier)

    return problem, objective


def prepare_to_add_variables(X):
    n_points = len(X)

    variable_assn_names = ["x_{}_{}".format(i, j) for i in range(n_points) for j in range(n_points)]
    variable_facility_names = ["y_{}".format(i) for i in range(n_points)]
    variable_names = variable_assn_names + variable_facility_names
    total_variables = n_points * n_points + n_points

    lower_bounds = [0] * total_variables
    upper_bounds = [1] * total_variables

    objective = cost_function(X)
    return objective, lower_bounds, upper_bounds, variable_names


def cost_function(X):
    dist = squareform(pdist(X, "sqeuclidean"))
    dist = dist.ravel().tolist()
    pad_for_facility = [0] * len(X)
    return dist + pad_for_facility


def prepare_to_add_constraints(X, n_clusters, Color, beta, alpha, violation):
    n_points = len(X)
    # The following steps constructs various types of constraints.
    sum_constraints, sum_rhs = constraint_sums_to_one(n_points)

    validity_constraints, validity_rhs = constraint_validity(n_points)

    facility_constraints, facility_rhs = constraint_facility(n_points, n_clusters)

    # We now combine all these types of constraints
    constraints_row = sum_constraints + validity_constraints + facility_constraints
    rhs = sum_rhs + validity_rhs + facility_rhs

    n_equality_constraints = len(sum_rhs)

    color_constraint, color_rhs = constraint_color(n_points, Color, beta, alpha, violation)
    constraints_row.extend(color_constraint)
    rhs.extend(color_rhs)

    n_inequality_constraints = len(rhs) - n_equality_constraints

    # The assignment constraints are of equality type and the rest are less than equal to type
    senses = ['E'] * n_equality_constraints + ['L'] * n_inequality_constraints
    # Name the constraints
    constraint_names = ["c_{}".format(i) for i in range(len(rhs))]

    return constraints_row, senses, rhs, constraint_names



def constraint_sums_to_one(n_points):
    constraints = [[["x_{}_{}".format(j,i) for i in range(n_points)], [1] * n_points] for j in range(n_points)]
    rhs = [1] * n_points
    return constraints, rhs


def constraint_validity(n_points):
    # This function adds the constraints of type: x_ji - y_i <= 0 for all i and j
    # [[['x_0_0', 'y_0'], [1, -1]],
    #  [['x_0_1', 'y_1'], [1, -1]],
    #  [['x_0_2', 'y_2'], [1, -1]],
    #  [['x_0_3', 'y_3'], [1, -1]],
    #  [['x_1_0', 'y_0'], [1, -1]],
    #  [['x_1_1', 'y_1'], [1, -1]],
    #  [['x_1_2', 'y_2'], [1, -1]],
    #  [['x_1_3', 'y_3'], [1, -1]]]
    constraints = [[["x_{}_{}".format(j,i),"y_{}".format(i)], [1,-1]] for j in range(n_points) for i in range(n_points)]
    rhs = [0] * (n_points * n_points)
    return constraints, rhs


def constraint_facility(n_points, n_clusters):
    # This function add the constraint sum_{i} y_i <= k
    # Assume there are 4 points.
    # [['y_1','y_2','y_3','y_4'][1,1,1,1]]
    constraints = [[["y_{}".format(i) for i in range(n_points)], [1]*n_points]]
    rhs = [n_clusters]
    return constraints, rhs


def constraint_color(n_points, Color, beta, alpha, violation):
    # this function adds the fairness constraint
    # the details are similar to the constraint_color function in the
    # cplex_Fair_Assignment_lp_solver file.
    beta_constraints = [[["x_{}_{}".format(j, i) for j in range(n_points)],
                         [beta[color] - 1 if Color[j] == color else beta[color] for j in range(n_points)]]
                        for i in range(n_points) for color, _ in beta.items()]
    alpha_constraints = [[["x_{}_{}".format(j, i) for j in range(n_points)],
                         [np.round(1 - alpha[color], decimals=3) if Color[j] == color else (-1) * alpha[color] for j in range(n_points)]]
                        for i in range(n_points) for color, _ in beta.items()]
    constraints = beta_constraints + alpha_constraints
    n_of_constraints = n_points * len(beta) * 2
    rhs = [violation] * n_of_constraints
    return constraints, rhs

if __name__ == "__main__":
    config_file = "config/example_config_1.ini"
    config = configparser.ConfigParser(converters={"list" : read_list})
    config.read(config_file)

    config_str = "bank"
    print("Using config = {}".format(config_str))

    data_dir = config[config_str].get("data_dir") # output/
    dataset = config_str # bank

    n_clusters = config[config_str].getint("n_clusters")
    delta = config[config_str].getfloat("delta")
    max_points = config[config_str].getint("max_points")
    violating = config["DEFAULT"].getboolean("violating")
    violation = config["DEFAULT"].getfloat("violation")

    fair_clustering(dataset, data_dir, n_clusters, delta, max_points, violating, violation)
