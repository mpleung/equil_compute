import numpy as np, pandas as pd, networkx as nx, itertools, sys, traceback

def assemble_data(data_folder):
    """
    We don't include a dummy for missings for data.church because Card and Guiliano do not report its coefficient. We don't include a dummy for missings for data.parent_HS, data.parent_college because Card and Giuliano do not report its coefficient. We only use one measure for physical development index because the other measures have too much missing data.
    """
    network_data = pd.read_csv(data_folder + '/21600-0003-Data.tsv', sep='\t', usecols=['AID','ODGX2'], low_memory=False)
    network_data.columns = ['id','outdeg']
    wave1 = pd.read_csv(data_folder + '/21600-0001-Data.tsv', sep='\t', usecols=['AID','H1MP4','H1EE14','H1EE15','S1','S2','S6B','S11','S12','S17','S18','PA22','PA23','PA63'], low_memory=False)
    wave1.columns = ['id','phys_dev','killed21','hiv','age','sex','black','has_mom','motheredu','has_dad','fatheredu','religious','church','HH_smoke']
    wave2 = pd.read_csv(data_folder + '/21600-0005-Data.tsv', sep='\t', usecols=['AID','H2PF28','H2PF35'], low_memory=False)
    wave2.columns = ['id','risk','future']

    data = pd.merge(wave1,wave2,on='id',how='inner')
    data = pd.merge(data,network_data,on='id',how='inner')
    data.replace(' ',np.nan,inplace=True)
    data.dropna(inplace=True)
    data = data.apply(pd.to_numeric)

    data = data[data.phys_dev < 6]
    data = data[data.risk < 6]
    data = data[data.future < 6]
    data = data[data.killed21 < 6]
    data = data[data.hiv < 6]
    data = data[data.HH_smoke <= 1]
    data = data[data.has_mom <= 1]
    data = data[data.has_dad <= 1]
    data = data[data.religious <= 28]
    data = data[data.church <= 4]
    data = data[data.age != 99]

    data['male'] = data.sex == 1
    data.phys_dev = (data.phys_dev - data.phys_dev.mean()) / data.phys_dev.std()
    data.risk.replace([1,2,3,4,5],[5,4,3,2,1], inplace=True)
    data['time_pref'] = (data.killed21 + data.hiv) / 2
    data['2parent'] = data.has_mom * data.has_dad
    data.religious = 1 - pd.eval('(data.religious == 28) or (data.church == 4)')
    data.church.replace([1,2,3,4],[3,2,1,0], inplace=True)
    data['parent_HS'] = pd.eval('((data.motheredu >=3) and (data.motheredu <= 8)) or ((data.fatheredu >=3) and (data.fatheredu <= 8))')
    data['parent_college'] = pd.eval('((data.motheredu >=7) and (data.motheredu <= 8)) or ((data.fatheredu >=7) and (data.fatheredu <= 8))')

    deg_dist = data.outdeg.values
    data = data[['age','black','male','phys_dev','risk','future','time_pref','HH_smoke','2parent','religious','church','parent_HS','parent_college']].values.astype(np.int32)

    if deg_dist.sum() % 2 != 0: deg_dist[0] += 1 # total degree must be even

    return deg_dist, data

def gen_exo(data, theta):
    """
    Outputs nx1 vector corresponding to X_i'beta, where X_i is the vector of covariates.
    """

    return data.dot(theta[4:theta.size])

def gen_D(U_exo_eps, A, theta):
    """
    Outputs the network D.

    U_exo_eps = output of gen_exo plus random-utility shock
    theta = numpy array; vector of structural parameters
    """
    D = nx.empty_graph(U_exo_eps.shape[0],create_using=nx.DiGraph())
    nr = (U_exo_eps - theta[0] + max(theta[2],0) > 0) * (U_exo_eps - theta[1] + min(theta[3],0) < 0) * (np.minimum(-U_exo_eps + theta[1] - max(theta[3],0), U_exo_eps - theta[0] + min(theta[2],0)) < 0)
    
    for edge in A.edges():
        i,j = edge[0], edge[1]
        if nr[j]: D.add_edge(i,j)
        if nr[i]: D.add_edge(j,i)

    return D

def component_NEs_one_run(component, rob12, nr_ind, A, U_exo_eps, theta, vec):
    """
    Outputs vec if it corresponds to a NE, empty list otherwise.

    component = list; agents in the component
    vec = element in itertools.product iterator; candidate action profile for agents in component with non-robust actions
    rob12 = numpy array; indicator for agents robustly choosing action 1
    nr_ind = numpy array; indicators for agents with non-robust actions
    A = networkx graph; network
    U_exo_eps = numpy array; vector of utilities, excluding endogenous part
    theta = numpy array; vector of structural parameters
    """
    # construct candidate action profile for the whole network, which consists of splicing vec into rob12
    nr_count = 0
    vec_list = list(vec)

    # verify equilibrium conditions for action_profile
    NE = True
    for i in component:
        Yneigh = np.array([vec_list[component.index(j)] for j in A.adj[i] if j in component] + [rob12[j] for j in A.adj[i] if j not in component]) # first list is actions of i's neighbors with non-robust actions (given by vec_list), second is actions of i's neighbors with robust actions (given by rob12)
        S1 = 0 if Yneigh.size == 0 else (Yneigh >= 1).sum() / Yneigh.size
        S2 = 0 if Yneigh.size == 0 else (Yneigh == 2).sum() / Yneigh.size
        c1 = theta[0] - theta[2] * S1
        c2 = theta[1] - theta[3] * S2
        own_action = vec_list[component.index(i)]
        if U_exo_eps[i] > c1 and own_action == 0 or (c1 > U_exo_eps[i] or U_exo_eps[i] > c2) and own_action == 1 or c2 > U_exo_eps[i] and own_action == 2:
            NE = False 
            break

    if NE:
        return vec_list
    else:
        return []

