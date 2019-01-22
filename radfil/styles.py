from matplotlib import rcParams
import numpy as np


#Set size opacity of scatter points based on total number of sampled points 
def get_scatter_alpha(numpoints):
    if numpoints < 1000:
        return 0.9
    
    elif numpoints < 10000:
        return 0.7
    
    elif numpoints < 100000:
        return 0.5
        
    else:
        return 0.1

## Add any matplotlib customizations of choice ##
rcParams['axes.titlesize'] = 20
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['legend.fontsize'] = 12
rcParams['axes.labelsize'] = 18