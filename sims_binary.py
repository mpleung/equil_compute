import numpy as np, pandas as pd, snap, os, sys, time
from scipy.stats import logistic
from functions_binary import *

np.random.seed(seed=0)

D_only = False # set to True to only simulate summary statistics of D

B = 100 # number of simulations

theta = np.array([-2.806, 0.64, -0.135, -0.034, 0.134, 0.064, 0.036, 1.717, -0.574, 0.043, 0.364, 1.052, -0.718, -1.098]) 
# structural parameters obtained from Xu (2018), Table 5, column AMLE(4)
# constant, peer effect, age, female, income, motheredu, fatheredu, gpa, native, asian, black, hispanic, white, other
theta[1] += 0.2
if D_only: theta[1] += 0.4 #0.8

### Assemble data ###

data_folder = 'data'
DegSeqV, data = assemble_data(data_folder)

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

    eps = np.random.logistic(size=U_exo.shape[0])
    U_exo_eps = U_exo + eps
    A = snap.GenConfModel(DegSeqV)

    start_time = time.time()
    D = gen_D(U_exo_eps, A, theta[1])
    components = snap.TCnComV()
    snap.GetSccs(D, components)
    component_lens = [C.Len() for C in components]
    print('Delta = {}.'.format(max(component_lens)))
    NE_sets = compute_NE(D, components, A, U_exo_eps, theta[1]) if not D_only else []
    end_time = time.time()
    timing[b] = end_time - start_time
    
    num_equil = 1
    total_takeup = []
    for i,C in enumerate(components):
        num = len(NE_sets[i]) if not D_only else 0
        if num == 0:
            num_equil = 0
            break
        else:
            num_equil *= num

        total_takeup.append([np.array(j).sum() for j in NE_sets[i]])

    num_NE[b] = num_equil
    D_giant[b] = max([C.Len() for C in components])
    A_giant[b] = snap.GetMxScc(A).GetNodes()
    D_deg[b] = D.GetEdges() / U_exo.shape[0]
    A_deg[b] = A.GetEdges() * 2 / U_exo.shape[0]
    takeup_bounds[b,0] = np.array([min(k) for k in total_takeup]).sum() / data.shape[0]
    takeup_bounds[b,1] = np.array([max(k) for k in total_takeup]).sum() / data.shape[0]

### Output ###

if not D_only:
    filename = 'results_binary.txt'
    if os.path.isfile(filename): os.remove(filename)
    f = open(filename, 'a')
    sys.stdout = f

U_exo_mean = theta[0] + data.mean(axis=0).dot(theta[2:theta.shape[0]])
print("Percentage marginal effect at mean covariates: {}".format((logistic.cdf(U_exo_mean + theta[1]*0.5) - logistic.cdf(U_exo_mean))/logistic.cdf(U_exo_mean)))

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

if not D_only: f.close()
