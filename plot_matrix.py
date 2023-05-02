import matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap

def plot_matrix(score_matrix):
    cmap=LinearSegmentedColormap.from_list('rg',["darkgreen","green" , "palegreen", "w","lightcoral", "red", "darkred"], N=256) 
    plt.xlabel("Probe")
    plt.ylabel("Gallery")
    plt.title("System_A Gallery and Probe Sets")
    plt.imshow(score_matrix, interpolation='none', cmap=cmap)
    plt.colorbar()
    plt.show()