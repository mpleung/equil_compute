import networkx as nx, numpy as np, pandas as pd, multiprocessing as mp, os, sys, time
from scipy.stats import norm
from functions_ordered import *
from functools import partial

np.random.seed(seed=0)
processes = 16 # numer of cores for parallelization
pool = mp.Pool(processes=processes)

B = 100 # number of simulations

theta = np.array([-1.5, 1.5, 0.2, 0.27, 0.06, 0.25, 0.04, 0.23, 0.11, -0.07, 0.1, 0.25, -0.2, -0.16, -0.06, -0.24, -0.1]) 
# Parameters correspond to: alpha_1, alpha_2, gamma_1, gamma_2, age, black, male, phys_dev, risk, future, time_pref, HH_smoke, 2parent, not religious, church, parent_HS, parent_college. (alphas and gammas defined in Example 7).
# Structural parameters (except the first two intercepts) obtained from Card and Giuliano (2013), Tables 3 and A3, both column 3.

### Assemble data ###

data_folder = 'data'
deg_dist, data = assemble_data(data_folder)

### Compute NE ###

timing = np.zeros(B)
D_giant = np.zeros(B)
A_giant = np.zeros(B)
D_deg = np.zeros(B)
A_deg = np.zeros(B)
num_NE = np.zeros(B)
takeup_bounds = np.zeros((B,2))

U_exo = gen_exo(data, theta)

for b in range(B):
    print(b)
    sys.stdout.flush()

    n = U_exo.shape[0]
    eps = np.random.normal(size=n)
    U_exo_eps = U_exo + eps
    A = nx.configuration_model(deg_sequence=deg_dist, seed=b)

    start_time = time.time()
    D = gen_D(U_exo_eps, A, theta)
    components = [C for C in nx.strongly_connected_components(D)]
    component_lens = [len(C) for C in components]
    print('Delta = {}.'.format(max(component_lens)))

    ### Compute NEs ###

    rob12 = (U_exo_eps - theta[1] + min(theta[3],0) > 0)*2 + (np.minimum(-U_exo_eps + theta[1] - max(theta[3],0), U_exo_eps - theta[0] + min(theta[2],0)) > 0)*1 # used as initial action profile, sets action of each agent to 1 or 2 if it's a robust action for her and otherwise sets action to 0
    nr_ind = (U_exo_eps - theta[0] + max(theta[2],0) > 0) * (U_exo_eps - theta[1] + min(theta[3],0) < 0) * (np.minimum(-U_exo_eps + theta[1] - max(theta[3],0), U_exo_eps - theta[0] + min(theta[2],0)) < 0) # indicator for non-robustness
    
    NE_sets = []

    # for each component, compute set of equilibria
    for C in components:
        component = [i for i in C]

        one_sim_wrapper = partial(component_NEs_one_run, component, rob12, nr_ind, A, U_exo_eps, theta)

        ### Exhaustive search ###

        m = np.array([nr_ind[i] for i in component]).sum() # number of agents in component with non-robust actions

        if m > 0:
            # iterate through \mathcal{Y}^*, checking equilibrium conditions
            if m < 12: # Delta is small, no need to parallelize
                equil_set = []
                for vec in itertools.product([0,1,2],repeat=m):
                    output = one_sim_wrapper(vec)
                    if len(output) > 0:
                        equil_set.append(output)
            else:
                results = pool.imap(one_sim_wrapper, itertools.product([0,1,2],repeat=m), chunksize=100000)
                equil_set = [r for r in results]
        else:
            # if there are no agents with non-robust actions, then components must consist of a singleton agent whose action is robust
            equil_set = [[rob12[component[0]]]]

        ### End exhaustive search ###

        equil_set = [s for s in equil_set if len(s) > 0] # set of equilibria on the component as a list of lists
        NE_sets.append(equil_set) # set of all equilibria

    ### End compute NEs ###

    end_time = time.time()
    timing[b] = end_time - start_time
    print('Time: {}'.format(timing[b]))
    
    num_equil = 1
    total_takeup = []
    for i,C in enumerate(components):
        num = len(NE_sets[i])
        if num == 0:
            num_equil = 0
            break
        else:
            num_equil *= num

        total_takeup.append([np.array(j).sum() for j in NE_sets[i]])

    num_NE[b] = num_equil
    D_giant[b] = max(component_lens)
    A_giant[b] = len(max(nx.connected_components(A), key=len))
    D_deg[b] = D.number_of_edges() / U_exo.shape[0]
    A_deg[b] = A.number_of_edges() * 2 / U_exo.shape[0]
    takeup_bounds[b,0] = np.array([min(k) for k in total_takeup]).sum() / data.shape[0]
    takeup_bounds[b,1] = np.array([max(k) for k in total_takeup]).sum() / data.shape[0]

pool.close()
pool.join()

### Output ###

filename = 'results_ordered.txt'
if os.path.isfile(filename): os.remove(filename)
f = open(filename, 'a')
sys.stdout = f

U_exo_mean = data.mean(axis=0).dot(theta[4:theta.size])
ME0 = (norm.cdf(-U_exo_mean + theta[0] - theta[2]*0.5) - norm.cdf(-U_exo_mean + theta[0]))/norm.cdf(-U_exo_mean + theta[0])
ME2 = (norm.cdf(U_exo_mean - theta[1] + theta[3]*0.5) - norm.cdf(U_exo_mean - theta[1]))/norm.cdf(U_exo_mean - theta[1])
print("Percentage marginal effects of choosing 0,2 at mean covariates: {},{}".format(ME0,ME2))

print('\n\\begin{table}[h]')
print('\centering')
print('\caption{Results}')
print('\\begin{threeparttable}')
table = pd.DataFrame(np.vstack([ \
        [takeup_bounds[:,0].mean(), takeup_bounds[:,0].std(), takeup_bounds[:,0].min(), takeup_bounds[:,0].max()], \
        [takeup_bounds[:,1].mean(), takeup_bounds[:,1].std(), takeup_bounds[:,1].min(), takeup_bounds[:,1].max()], \
        [num_NE.mean(), num_NE.std(), num_NE.min(), num_NE.max()], \
        [timing.mean(), timing.std(), timing.min(), timing.max()], \
        [D_giant.mean(), D_giant.std(), D_giant.min(), D_giant.max()], \
        [D_deg.mean(), D_deg.std(), D_deg.min(), D_deg.max()], \
        [A_giant.mean(), A_giant.std(), A_giant.min(), A_giant.max()], \
        [A_deg.mean(), A_deg.std(), A_deg.min(), A_deg.max()], \
        ]).T)
table.index = ['Mean', 'SD', 'Min', 'Max']
table.columns = ['$\\bar{Y}$ Lower', '$\\bar{Y}$ Upper', '\# NE', 'Time', '$\\Delta$', '$\\bm{D}$ Deg', '$\\bm{A}$ Giant', '$\\bm{A}$ Deg']
print(table.to_latex(float_format = lambda x: '%.3f' % x, header=True, escape=False))
print('\\begin{tablenotes}[para,flushleft]')
print('  \small $n={}$, 100 simulations. Column 1 is the smallest average outcome across simulations, column 2 the largest. Column 3 is the number of equilibria. Column 4 is computation time in seconds. Column 5 (7) is the size of the largest component of $\\bm{{D}}$ ($\\bm{{A}}$) and column 6 (8) the average degree of $\\bm{{D}}$ ($\\bm{{A}}$).'.format(U_exo.shape[0]))
print('\end{tablenotes}')
print('\end{threeparttable}')
print('\end{table}')

f.close()
