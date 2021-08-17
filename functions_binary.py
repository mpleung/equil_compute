import numpy as np, pandas as pd, snap, itertools

def assemble_data(data_folder):
    network_data = pd.read_csv(data_folder + '/21600-0003-Data.tsv', sep='\t', usecols=['AID','ODGX2'])
    network_data.columns = ['id','outdeg']
    covariate_data = pd.read_csv(data_folder + '/21600-0001-Data.tsv', sep='\t', usecols=['AID','S1','S2','S4','S6A','S6B','S6C','S6D','S6E','H1ED11','H1ED12','H1ED13','H1ED14','S12','S18','PA55'])
    covariate_data.columns = ['id','english','math','history','science','age','sex','hispanic','white','black','asian','native','other','motheredu','fatheredu','income']

    data = pd.merge(network_data,covariate_data,on='id',how='inner')
    data.replace(' ',np.nan,inplace=True)
    data.dropna(inplace=True)
    data = data.apply(pd.to_numeric)

    data = data[data.age != 99]
    data = data[data.income != 9996]
    data = data[data.math < 6]
    data = data[data.english < 6]
    data = data[data.science < 6]
    data = data[data.history < 6]
    data = data[data.motheredu < 97]
    data = data[data.motheredu != 9]
    data = data[data.fatheredu < 97]
    data = data[data.fatheredu != 9]
    data['female'] = data.sex == 2
    data.hispanic = data.hispanic == 1
    data['gpa'] = ( 4*((data.math==1)+(data.science==1)+(data.english==1)+(data.history==1)) + 3*((data.math==2)+(data.science==2)+(data.english==2)+(data.history==2)) + 2*((data.math==3)+(data.science==3)+(data.english==3)+(data.history==3)) + ((data.math==4)+(data.science==4)+(data.english==4)+(data.history==4)) ) / float(4)
    data.motheredu = (data.motheredu==10) + (data.motheredu==11) + 2*(data.motheredu==1) + 2*(data.motheredu==2) + 2*(data.motheredu==4) + 3*(data.motheredu==3) + 3*(data.motheredu==5) + 3*(data.motheredu==6) + 4*(data.motheredu==7) + 5*(data.motheredu==8)
    data.fatheredu = (data.fatheredu==10) + (data.fatheredu==11) + 2*(data.fatheredu==1) + 2*(data.fatheredu==2) + 2*(data.fatheredu==4) + 3*(data.fatheredu==3) + 3*(data.fatheredu==5) + 3*(data.fatheredu==6) + 4*(data.fatheredu==7) + 5*(data.fatheredu==8)
    brackets = [np.percentile(data.income,i) for i in np.linspace(0,100,13)]
    data.income = (data.income>brackets[0])*(data.income<brackets[1]) + 2*(data.income>brackets[1])*(data.income<brackets[2]) + 3*(data.income>brackets[2])*(data.income<brackets[3]) + 4*(data.income>brackets[3])*(data.income<brackets[4]) + 5*(data.income>brackets[4])*(data.income<brackets[5]) + 6*(data.income>brackets[5])*(data.income<brackets[6]) + 7*(data.income>brackets[6])*(data.income<brackets[7]) + 8*(data.income>brackets[7])*(data.income<brackets[8]) + 9*(data.income>brackets[8])*(data.income<brackets[9]) + 10*(data.income>brackets[9])*(data.income<brackets[10]) + 11*(data.income>brackets[10])*(data.income<brackets[11]) + 12*(data.income>brackets[11])*(data.income<brackets[12])

    deg_dist = data['outdeg'].values
    data = data[['age','female','income','motheredu','fatheredu','gpa','native','asian','black','hispanic','white','other']].values.astype(np.int32)

    # create degree distribution
    if deg_dist.sum() % 2 != 0: deg_dist[0] += 1 # total degree must be even
    DegSeqV = snap.TIntV()
    for i in deg_dist:
        DegSeqV.Add(i)

    return DegSeqV, data

def gen_exo(data, theta):
    """
    Outputs nx1 vector consisting of exogenous part of utility (with no random utility shocks).
    """
    return theta[0] + data.dot(theta[2:theta.shape[0]])

def gen_D(U_exo_eps, A, beta):
    """
    Outputs the network D.

    U_exo_eps = output of gen_exo plus random-utility shocks.
    beta = peer effect parameter.
    """
    D = snap.GenRndGnm(snap.PNGraph, U_exo_eps.shape[0], 0)
    nr = (U_exo_eps + min(beta,0) < 0) * (U_exo_eps + max(beta,0) > 0)
    
    for edge in A.Edges():
        i = min(edge.GetSrcNId(), edge.GetDstNId())
        j = max(edge.GetSrcNId(), edge.GetDstNId())
        if nr[j]: D.AddEdge(i,j)
        if nr[i]: D.AddEdge(j,i)

    return D

def compute_NE(D, D_components, A, U_exo_eps, beta):
    """
    Outputs tuple. First element is a vector of components in the form of a snap TCnComV vector. The second is a list of lists, each corresponding to the output of component_NEs for a given component.

    D = snap.PNGraph; output of gen_D()
    D_components = strongly connected components of D, stored in a snap.TCnComV() object
    A = snap.PUNGraph; network
    U_exo_eps = numpy array; vector of utilities, excluding endogenous part
    beta = scalar; endogenous peer effect
    """
    n = U_exo_eps.shape[0]

    rob1 = (U_exo_eps + min(beta,0) > 0)*1 # used as initial action profile, which sets action of each agent to 1 if that's a robust action for her and otherwise sets action to 0
    nr_ind = (1-rob1)*(U_exo_eps + max(beta,0) > 0) # indicator for non-robust actions

    all_component_NEs = []
    NIdV = snap.TIntV() 

    # for each component, get set of equilibria
    for C in D_components:
        NIdV.Clr()
        for i in C:
            NIdV.Add(i)
        all_component_NEs.append(component_NEs(NIdV, A, U_exo_eps, rob1, nr_ind, beta))

    return all_component_NEs

def component_NEs(component, A, U_exo_eps, rob1, nr_ind, beta):
    """
    Outputs list of lists. Each list corresponds to the equilibrium set \mathcal{E}(T_{\mathcal{S}(C)}, A_{\mathcal{S}(C)})|_C for a component C.

    component = snap vector; vector of agents in the component
    A = snap.PUNGraph; network
    U_exo_eps = numpy array; vector of utilities, excluding endogenous part
    rob1 = numpy array; indicator for agents robustly choosing action 1
    nr_ind = numpy array; indicators for agents with non-robust actions
    beta = scalar; endogenous peer effect
    """
    n = component.Len()
    m = np.array([nr_ind[i] for i in component]).sum() # number of agents in component with non-robust actions

    if m > 0:
        equil_set = []

        # iterate through \mathcal{Y}^*, checking equilibrium conditions
        for vec in itertools.product([0,1],repeat=m):
            output = component_NEs_one_run(component, list(vec), rob1.copy(), nr_ind, A, U_exo_eps, beta)
            if len(output) > 0:
                equil_set.append(output)
    else:
        # if there are no agents with non-robust actions, then each component must consist of a singleton agent whose action is robust
        equil_set = [[rob1[component[0]]]]

    return equil_set

def component_NEs_one_run(component, vec, action_profile, nr_ind, A, U_exo_eps, beta):
    """
    Outputs vec if it induces a NE, empty list otherwise.

    vec = list; candidate action profile for agents in component with non-robust actions
    action_profile = numpy array; deep copy of rob1
    """
    # construct candidate action profile for the whole network, which consists of splicing vec into rob1
    nr_count = 0
    for i in component:
        if nr_ind[i]:
            action_profile[i] = vec[nr_count]
            nr_count += 1

    # verify equilibrium conditions for action_profile
    NE = True
    for i in component:
        Yneigh = np.array([action_profile[j] for j in A.GetNI(i).GetOutEdges()])
        Si = 0 if Yneigh.size == 0 else Yneigh.sum() / Yneigh.size
        if U_exo_eps[i] + beta*Si > 0 and action_profile[i] == 0 or U_exo_eps[i] + beta*Si < 0 and action_profile[i] == 1:
            NE = False
            break
    
    if NE:
        return vec
    else:
        return []
