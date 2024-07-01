import numpy as np
import matplotlib.pyplot as plt

# Color coated board
colors = ["dodgerblue", "darkorange", "forestgreen", "darkviolet", "deeppink",
          "gold", "darkcyan", "sienna", "peru", "violet", "blue", "lightcoral",
          "limegreen", "olive", "y", "royalblue", "tomato", "cadetblue",
          "blueviolet", "crimson", "chocolate", "orangered", "tan", "navy"]

def ShowClustering(X, y, centers = None, Dataset = "Null", Algorithm = "Null", saveFlag = False):
    # Set figure size
    figWidth = 6
    fig = plt.figure(figsize=(figWidth, figWidth))
    # Set background color
    fig.patch.set_facecolor('white')
    # Set Transparency
    fig.patch.set_alpha(1)
    # Get axes()
    current_axes = plt.axes()
    current_axes.get_xaxis().set_visible(False)
    current_axes.get_yaxis().set_visible(False)
    # Data size
    DataSize = len(y)
    # Point size
    if DataSize <= 250:
        PointSize = 40
    elif DataSize <= 350:
        PointSize = 35
    elif DataSize <= 500:
        PointSize = 30
    elif DataSize <= 600:
        PointSize = 28
    elif DataSize <= 800:
        PointSize = 26
    elif DataSize <= 1000:
        PointSize = 20
    else:
        PointSize = 15
    # Label catalogue
    Labels = np.unique(y)
    ColorIndex, CL = 0, len(colors)
    for label in Labels:
        indeces = (y == label)
        plt.scatter(X[indeces, 0], X[indeces, 1], c=colors[ColorIndex % CL], s=PointSize)
        ColorIndex += 1
    if centers is not None:
        centers = np.array(centers)
        plt.scatter(X[centers, 0], X[centers, 1], c="red", s=300, marker="*")
    # Save figure
    if saveFlag:
        title = "../Figures/Figure_exp_" + Algorithm + "_" + Dataset
        plt.savefig(title + ".pdf", dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.savefig(title + ".eps", dpi=400, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close()