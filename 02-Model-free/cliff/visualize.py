import matplotlib.pyplot as plt
import numpy as np


def cm2inch(*tupl):
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def heatmap(AUC):
    # Plot it out
    fig, ax = plt.subplots()
    c = ax.pcolor(AUC, edgecolors='k', linestyle='dashed', linewidths=1, cmap='YlOrRd', vmin=0.0, vmax=2.0)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # Remove last blank column
    #plt.xlim((0, AUC.shape[1]))

    # resize
    #fig = plt.gcf()
    fig.set_size_inches(cm2inch(40, 20))


def render(data):
    heatmap(data)
    # plt.savefig('image_output.png', dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures
    plt.show()
