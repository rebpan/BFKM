from gurobipy import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
import copy

def data_load(name):
    if name == "adult":
        X = np.loadtxt(r"../../Datasets/subsampled_adult.txt", delimiter=",")
        Color = np.loadtxt(r"../../Datasets/subsampled_adult_Color.txt", dtype=int)
        K = 5
    elif name == "bank":
        X = np.loadtxt(r"../../Datasets/subsampled_bank.txt", delimiter=",")
        Color = np.loadtxt(r"../../Datasets/subsampled_bank_Color.txt", dtype=int)
        K = 6
    elif name == "census1990":
        X = np.loadtxt(r"../../Datasets/subsampled_census1990.txt", delimiter=",")
        Color = np.loadtxt(r"../../Datasets/subsampled_census1990_Color.txt", dtype=int)
        K = 5
    elif name == "creditcard":
        X = np.loadtxt(r"../../Datasets/subsampled_creditcard.txt", delimiter=",")
        Color = np.loadtxt(r"../../Datasets/subsampled_creditcard_Color.txt", dtype=int)
        K = 5
    elif name == "diabetic":
        X = np.loadtxt(r"../../Datasets/subsampled_diabetic.txt", delimiter=",")
        Color = np.loadtxt(r"../../Datasets/subsampled_diabetic_Color.txt", dtype=int)
        K = 10
    elif name == "elliptical":
        X = np.loadtxt(r"../../Datasets/elliptical.txt", delimiter=",")
        Color = np.loadtxt(r"../../Datasets/elliptical_Color.txt", dtype=int)
        K = 2
    elif name == "DS577":
        X = np.loadtxt(r"../../Datasets/DS577.txt", delimiter=",")
        Color = np.loadtxt(r"../../Datasets/DS577_Color.txt", dtype=int)
        K = 3
    elif name == "2d-4c-no0":
        X = np.loadtxt(r"../../Datasets/2d-4c-no0.txt", delimiter=",")
        Color = np.loadtxt(r"../../Datasets/2d-4c-no0_Color.txt", dtype=int)
        K = 4
    elif name == "2d-4c-no1":
        X = np.loadtxt(r"../../Datasets/2d-4c-no1.txt", delimiter=",")
        Color = np.loadtxt(r"../../Datasets/2d-4c-no1_Color.txt", dtype=int)
        K = 4
    elif name == "2d-4c-no4":
        X = np.loadtxt(r"../../Datasets/2d-4c-no4.txt", delimiter=",")
        Color = np.loadtxt(r"../../Datasets/2d-4c-no4_Color.txt", dtype=int)
        K = 4
    else:
        return
    return X, Color, K
        
if __name__ == "__main__":
    name = "2d-4c-no4"
    X_raw, Color, K = data_load(name)
    # standardization and normalization
    X = scale(X_raw, axis = 0)
    feanorm = np.maximum(1e-14,np.sum(X**2,axis=1))
    X = X/(feanorm[:,None]**0.5)

    # Vanilla K-means Clustering
    kmeans = KMeans(n_clusters = K, random_state = 0)
    y_kmeans = kmeans.fit_predict(X)
    S=kmeans.cluster_centers_

    # Processing Color
    uq_color = np.unique(Color)
    attribute = {uq_color[k] : list(np.where(Color == uq_color[k])[0]) for k in range(len(uq_color))}
    representation = {uq_color[k] : len(attribute[k])/len(Color) for k in range(len(uq_color))}

    # Fairness Bounds
    delta = 0.2
    a_val, b_val = 1 / (1 - delta), 1 - delta
    alpha = {uq_color[k] : a_val * representation[k] for k in range(len(uq_color))}
    beta = {uq_color[k] : b_val * representation[k] for k in range(len(uq_color))}

    # LP1
    model=Model("LP1")

    n_points = len(X)
    x={}
    for v in range(n_points):
        for f in range(K):
            x[v,f]=model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1)

    for v in range(n_points):
        model.addConstr(quicksum(x[v,f] for f in range(K))==1)

    for f in range(K):
        for i in attribute.keys():
            model.addConstr(quicksum(x[v,f] for v in attribute[i]) <= alpha[i]*quicksum(x[v,f] for v in range(n_points)))
            model.addConstr(quicksum(x[v,f] for v in attribute[i]) >= beta[i]*quicksum(x[v,f] for v in range(n_points)))

    Obj=0
    for v in range(n_points):
        for f in range(K):
            Obj+=x[v,f]*(np.linalg.norm(X[v]-S[f])**2)

    model.setObjective(Obj)
    model.optimize()

    # Assigning v with x*[v,f]=1 to final_asst, the dictionary containing the final fair clustered points
    x_star={}
    for v in range(n_points):
        for f in range(K):
            x_star[v,f]=x[v,f].X

    final_asst = {f: [] for f in range(K)}
    for v in range(n_points):
        for f in range(K):
            if x_star[v,f]>=1:
                final_asst[f].append(v)

    assigned = []
    for f in range(K):
        assigned.extend(final_asst[f])
    diff_set = list(set(list(range(n_points))) ^ set(assigned))

    # Defining dictionaries T_f and T_f_i
    T_f={f:0 for f in range(K)}
    for f in range(K):
        for v in range(n_points):
            T_f[f]+= x_star[v,f]

    T_f_i = {(f,i) : 0 for f in range(len(S)) for i in attribute.keys()}
    for f in range(K):
        for i in attribute.keys():
            for j in attribute[i]:
                T_f_i[f,i] += x_star[j,f]

    # LP2 solved iteratively, until all variables are assigned to a cluster with fairness considerations
    diff_set_cp = copy.copy(diff_set)

    while(len(diff_set)>0):
        model=Model("LP2")

        x={}
        for v in range(n_points):
            for f in range(K):
                if x_star[v,f]>0:
                    x[v,f]=model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1)

        for f in range(len(S)):
            model.addConstr(quicksum(x[v,f] for v in range(n_points) if x_star[v,f]>0) <= np.ceil(T_f[f]))
            model.addConstr(quicksum(x[v,f] for v in range(n_points) if x_star[v,f]>0) >= np.floor(T_f[f]))

        for f in range(len(S)):
            for i in list(attribute.keys()):
                model.addConstr(quicksum(x[v,f] for v in attribute[i] if x_star[v,f]>0) <= np.ceil(T_f_i[f,i]))
                model.addConstr(quicksum(x[v,f] for v in attribute[i] if x_star[v,f]>0) >= np.floor(T_f_i[f,i]))

        for v in diff_set:
            model.addConstr(quicksum(x[v,f] for f in range(K) if x_star[v,f]>0)==1)

        Obj=0
        for v in diff_set:
            for f in range(K):
                if x_star[v,f]>0:
                    Obj+=x[v,f]*(np.linalg.norm(X[v]-S[f])**2)

        model.setObjective(Obj)
        model.optimize()

        for v in diff_set:
            for f in range(K):
                if (x_star[v,f]>0) and (x[v,f].X==1.0):
                    diff_set.remove(v)

    for v in diff_set_cp:
        for f in range(K):
            if (x_star[v,f]>0) and (x[v,f].X==1.0):
                final_asst[f].append(v)

    assigned = []
    for f in range(K):
        assigned.extend(final_asst[f])
    diff_set = list(set(list(range(n_points))) ^ set(assigned))

    # convert dict to ndarray
    label = np.zeros(n_points, dtype=int)
    for f in final_asst.keys():
        label[final_asst[f]] = f

    if name == "adult":
        np.savetxt("adult_FAC.txt", label+1, fmt="%d")
    elif name == "bank":
        np.savetxt("bank_FAC.txt", label+1, fmt="%d")
    elif name == "census1990":
        np.savetxt("census1990_FAC.txt", label+1, fmt="%d")
    elif name == "creditcard":
        np.savetxt("creditcard_FAC.txt", label+1, fmt="%d")
    elif name == "diabetic":
        np.savetxt("diabetic_FAC.txt", label+1, fmt="%d")
    elif name == "elliptical":
        np.savetxt("elliptical_FAC.txt", label+1, fmt="%d")
    elif name == "DS577":
        np.savetxt("DS577_FAC.txt", label+1, fmt="%d")
    elif name == "2d-4c-no0":
        np.savetxt("2d-4c-no0_FAC.txt", label+1, fmt="%d")
    elif name == "2d-4c-no1":
        np.savetxt("2d-4c-no1_FAC.txt", label+1, fmt="%d")
    elif name == "2d-4c-no4":
        np.savetxt("2d-4c-no4_FAC.txt", label+1, fmt="%d")
    else:
        pass
