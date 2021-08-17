This repository contains replication files for "Equilibrium Computation in Discrete Network Games". The files were coded for Python 3 and require the following dependencies: numpy, scipy, networkx, pandas, and [snap](https://snap.stanford.edu/snappy/index.html).

Contents:
* sims\_binary.py: binary choice numerical illustration.
* functions\_binary.py: functions used in sims\_binary.py. 
* sims\_ordered.py: ordered choice numerical illustration.
* functions\_ordered.py: functions used in sims\_ordered.py.

To run the code, download public-use Add Health data from [ICPSR](https://www.icpsr.umich.edu/icpsrweb/ICPSR/studies/21600?archive=ICPSR&q=21600), extract the contents, and then place the following data files into a folder called 'data' in the same directory as the Python code: 21600-0001-Data.tsv, 21600-0003-Data.tsv, 21600-0005-Data.tsv. 

Note that sims\_ordered.py uses 16 parallel processes, which can be changed by specifying a different value of the 'processes' variable at the top of the code.

