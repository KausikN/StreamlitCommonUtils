# Imports
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Main Functions
# Plot Custom Dataset Functions
def plot_labelled_data(Dataset, title="", plot=False) -> np.ndarray:
    '''
    Plots the data with labels

    Args:
        Dataset (dict): A dictionary containing the dataset to plot
        title (str): The title of the plot
        plot (bool): Whether to show the plot

    Returns:
        I_plot (array-like): The plotted image as a numpy array
    '''
    fig = plt.figure()
    canvas = FigureCanvasAgg(fig)
    fig.clear()

    X = Dataset["points"]
    labels = Dataset["labels"]
    unique_labels = Dataset["unique_labels"]
    centers = np.array([])
    if "centers" in Dataset.keys():
        centers = Dataset["centers"]

    # Init Plot
    ax = None
    # If 3D
    if Dataset["dim"] >= 3:
        ax = plt.axes(projection="3d")
    else:
        ax = plt.axes()

    # Plot the data
    for ul in unique_labels:
        X_ul = X[labels == ul]

        if Dataset["dim"] >= 3:
            ax.scatter3D(X_ul[:, 0], X_ul[:, 1], X_ul[:, 2], label=ul)
        elif Dataset["dim"] == 2:
            ax.scatter(X_ul[:, 0], X_ul[:, 1], label=ul)
        elif Dataset["dim"] == 1:
            ax.scatter(X_ul[:, 0], np.zeros(X_ul.shape), label=ul)

    # Plot Centers
    if centers.shape[0] > 0:
        centersParams = {"marker": "x", "label": "Centers", "s": 50, "c": "black"}
        if Dataset["dim"] >= 3:
            ax.scatter3D(centers[:, 0], centers[:, 1], centers[:, 2], **centersParams)
        elif Dataset["dim"] == 2:
            ax.scatter(centers[:, 0], centers[:, 1], **centersParams)
        elif Dataset["dim"] == 1:
            ax.scatter(centers[:, 0], np.zeros(centers.shape), **centersParams)

    plt.legend()
    plt.title(title)

    canvas.draw()
    buf = canvas.buffer_rgba()
    I_plot = cv2.cvtColor(np.asarray(buf), cv2.COLOR_RGBA2RGB)
    
    if plot: plt.show()
    plt.close(fig)

    return I_plot

def plot_unlabelled_data(Dataset, title="", lines=True, plot=False) -> np.ndarray:
    '''
    Plots the datapoints

    Args:
        Dataset (dict): A dictionary containing the dataset to plot
        title (str): The title of the plot
        lines (bool): Whether to connect the points with lines
        plot (bool): Whether to show the plot

    Returns:
        I_plot (array-like): The plotted image as a numpy array
    '''
    fig = plt.figure()
    canvas = FigureCanvasAgg(fig)
    fig.clear()

    X = Dataset["points"]
    centers = np.array([])
    if "centers" in Dataset.keys():
        centers = Dataset["centers"]

    # Init Plot
    ax = None
    # If 3D
    if Dataset["dim"] >= 3:
        ax = plt.axes(projection="3d")
    else:
        ax = plt.axes()

    # Plot the data
    if Dataset["dim"] >= 3:
        if not lines:
            ax.scatter3D(X[:, 0], X[:, 1], X[:, 2])
        else:
            ax.plot3D(X[:, 0], X[:, 1], X[:, 2])
    elif Dataset["dim"] == 2:
        if not lines:
            ax.scatter(X[:, 0], X[:, 1])
        else:
            ax.plot(X[:, 0], X[:, 1])
    elif Dataset["dim"] == 1:
        if not lines:
            ax.scatter(X[:, 0], np.zeros(X.shape))
        else:
            ax.plot(X[:, 0], np.zeros(X.shape))

    # Plot Centers
    if centers.shape[0] > 0:
        centersParams = {"marker": "x", "label": "Centers", "s": 50, "c": "black"}
        if Dataset["dim"] >= 3:
            ax.scatter3D(centers[:, 0], centers[:, 1], centers[:, 2], **centersParams)
        elif Dataset["dim"] == 2:
            ax.scatter(centers[:, 0], centers[:, 1], **centersParams)
        elif Dataset["dim"] == 1:
            ax.scatter(centers[:, 0], np.zeros(centers.shape), **centersParams)

    # plt.legend()
    plt.title(title)

    canvas.draw()
    buf = canvas.buffer_rgba()
    I_plot = cv2.cvtColor(np.asarray(buf), cv2.COLOR_RGBA2RGB)
    
    if plot:
        plt.show()
    plt.close(fig)

    return I_plot

# Plot NetworkX Graph Functions
def plot_networkx_graph(
    G, colors="#1f78b4", 
    show_edge_wt=False, title="", pos=None
) -> np.ndarray:
    '''
    Plot Graph - Plots a NetworkX graph
    '''
    fig = plt.figure()
    canvas = FigureCanvasAgg(fig)
    fig.clear()

    # Init Plot
    ax = plt.axes()
    
    # Plot Graph
    if pos is None:
        # pos = nx.spring_layout(G)
        pos = nx.circular_layout(G)
        # pos = nx.multipartite_layout(G)
        # pos = nx.spectral_layout(G)
        # pos = nx.shell_layout(G)
        # pos = nx.planar_layout(G)
        # pos = nx.nx_agraph.graphviz_layout(G)

    nx.draw_networkx(G, pos, ax=ax, with_labels=True, node_color=colors, font_color=(0.3, 1.0, 0.3))

    if show_edge_wt:
        labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, font_size=7, edge_labels=labels, ax=ax)

    plt.title(title)
    ax.set_facecolor((0.8, 0.6, 0.6))

    canvas.draw()
    buf = canvas.buffer_rgba()
    I_plot = cv2.cvtColor(np.asarray(buf), cv2.COLOR_RGBA2RGB)
    plt.close(fig)

    return I_plot, pos

# Plot Matplotlib Animations
def create_matplotlib_animation(
    plotFig, updateFunc, initFunc, 
    frames=np.linspace(0, 2*np.pi, 64), show=False
) -> FuncAnimation:
    '''
    Create a matplotlib animation from a plot figure, update function, init function and frames

    Args:
        plotFig: The matplotlib figure to animate
        updateFunc: The function that updates the plot for each frame
        initFunc: The function that initializes the plot
        frames: The frames for the animation (default is a linspace from 0 to 2*pi)

    Returns:
        FuncAnimation: The created matplotlib animation
    '''
    animation = FuncAnimation(plotFig, updateFunc, frames, init_func=initFunc)
    if show: plt.show()
    return animation

# Plot Values List Functions
def plot_values_list(
    values, titles=["", "", ""], 
    plotLines=True, plotPoints=True, annotate=False, plot=True
) -> np.ndarray:
    '''
    List - Visualise a list of values as a plot

    Args:
        values (list): The list of values to plot
        titles (list): A list of 3 strings for the x-label, y-label and title of the plot
        plotLines (bool): Whether to connect the points with lines
        plotPoints (bool): Whether to plot the points
        annotate (bool): Whether to annotate the points with their values
        plot (bool): Whether to show the plot

    Returns:
        I_plot (array-like): The plotted image as a numpy array
    '''
    fig, ax = plt.subplots()
    canvas = FigureCanvasAgg(fig)
    if plotLines:
        ax.plot(list(range(1, len(values)+1)), values)
    if plotPoints:
        ax.scatter(list(range(1, len(values)+1)), values)
    plt.xlabel(titles[0])
    plt.ylabel(titles[1])
    plt.title(titles[2])
    values_str = []
    for i in range(len(values)):
        values_str.append(str(values[i]))
        if annotate:
            ax.annotate(str(values[i]), (i+1, values[i]))
    
    if plot: plt.show()

    canvas.draw()
    buf = canvas.buffer_rgba()
    I_plot = np.asarray(buf)

    return I_plot