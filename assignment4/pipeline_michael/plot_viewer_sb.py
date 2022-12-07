# Script to view different seaborn plots saved as pickle. 
# make command line interface in next step


import pickle
import seaborn as sb
import matplotlib.pyplot as plt
figx = pickle.load(open("./figures/seaborn/f0_2_t6d.pickle", 'rb'))

plt.show() # Show the figure, edit it, etc.!

