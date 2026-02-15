# Imports
import json
import numpy as np
from typing import Dict
from sklearn.datasets import make_blobs

# Main Functions
# Generate Functions
# Points Datasets
def generate_random_blobs(N, dim, centers) -> Dict[str, object]:
    '''
    Generates a random dataset of 2D points

    Args:
        N (int): Number of samples
        dim (int): Number of dimensions
        centers (int): Number of centers to generate

    Returns:
        Dataset (dict): A dictionary containing the generated dataset
    '''
    # Init Dataset
    Dataset = {}
    # Generate random dataset
    X, y = make_blobs(n_samples=N, n_features=dim, centers=centers, random_state=42)
    Dataset["points"] = np.array(X)
    Dataset["labels"] = np.array(y)
    Dataset["unique_labels"] = np.unique(Dataset["labels"])
    Dataset["dim"] = dim

    return Dataset

def generate_points_from_image(I) -> Dict[str, object]:
    '''
    Generates a dataset of points from an image

    Args:
        I (array-like): Input image (2D array)
    
    Returns:
        Dataset (dict): A dictionary containing the generated dataset
    '''
    # Init Dataset
    Dataset = {}
    # Assign binary image
    I_bin = np.array(I)
    # Get points
    points = list(zip(*np.where(I_bin)))
    points = np.array(points)
    Dataset["points"] = points
    Dataset["labels"] = np.zeros(points.shape[0], dtype=int)
    Dataset["unique_labels"] = np.unique(Dataset["labels"])
    Dataset["dim"] = 2

    return Dataset

def generate_polynomial_dist_data(
    N, x_dim, y_dim, 
    valRange=[-1.0, 1.0]
) -> Dict[str, object]:
    '''
    Generates a dataset of Xs with Y = poly(X)

    Args:
        N (int): Number of samples
        x_dim (int): Number of dimensions for X
        y_dim (int): Number of dimensions for Y
        valRange (list): Range of values for X

    Returns:
        Dataset (dict): A dictionary containing the generated dataset
    '''
    # Init Dataset
    Dataset = {}
    # Generate random dataset
    X = np.random.uniform(valRange[0], valRange[1], (N, x_dim))

    Ys = []
    for i in range(y_dim):
        randomPolyDegree = np.random.randint(-5, 6)
        randomCoeffs = np.random.uniform(-1.0, 1.0, (x_dim + 1))
        Y = np.sum(randomCoeffs[1:] * (X**randomPolyDegree), axis=-1) + randomCoeffs[0]
        Ys.append(Y)
    Ys = np.dstack(Ys)[0]
    Dataset["X"] = np.array(X)
    Dataset["Y"] = np.array(Ys)
    Dataset["X_dim"] = x_dim
    Dataset["Y_dim"] = y_dim

    return Dataset

def generate_polynomial_noisy_data_2D(
    N, degree, 
    noise_factor=0.5, valRange=[-1.0, 1.0], coeffValRange=[-1.0, 1.0]
) -> Dict[str, object]:
    '''
    Generates a dataset of Xs with Y = poly(X) with noise

    Args:
        N (int): Number of samples
        degree (int): Degree of the polynomial
        noise_factor (float): Standard deviation of the Gaussian noise
        valRange (list): Range of values for X
        coeffValRange (list): Range of values for polynomial coefficients

    Returns:
        Dataset (dict): A dictionary containing the generated dataset
    '''
    # Init Dataset
    Dataset = {}
    # Generate random dataset
    X = np.random.uniform(valRange[0], valRange[1], N)

    randomCoeffs = np.random.uniform(coeffValRange[0], coeffValRange[1], (degree + 1))
    Y = np.zeros(N)
    for i in range(degree+1):
        Y += randomCoeffs[i] * (X**i) + np.random.normal(0, noise_factor, N)

    Dataset["X"] = np.array(X)
    Dataset["Y"] = np.array(Y)
    Dataset["degree"] = degree

    return Dataset

# Graph Datasets
def generate_adjacency_matrix_random(
    N, prob_edge=0.5, weight_range=[0.1, 1.0], 
    weights_int_only=False, no_self_loops=True, undirected=True
) -> np.ndarray:
    '''
    Generates a random adjacency matrix

    Args:
        N (int): Number of nodes
        prob_edge (float): Probability of edge existence
        weight_range (list): Range of edge weights
        weights_int_only (bool): Whether to round weights to integers
        no_self_loops (bool): Whether to disallow self-loops
        undirected (bool): Whether the graph is undirected

    Returns:
        Adj (array): Generated adjacency matrix
    '''
    # Generate random adjacency matrix
    Adj = np.random.uniform(low=weight_range[0], high=weight_range[1], size=(N, N))
    Adj_mask = np.random.uniform(low=0.0, high=1.0, size=(N, N)) > prob_edge
    Adj[Adj_mask] = np.inf
    if weights_int_only:
        Adj = np.round(Adj, decimals=0)
    if no_self_loops:
        Adj[np.diag_indices(N)] = np.inf
    if undirected:
        for i in range(N):
            for j in range(N):
                Adj[i, j] = min(Adj[i, j], Adj[j, i])
    
    return Adj

def generate_json_data_from_adjacency_matrix(Adj) -> str:
    '''
    Generates jsonData from a adjacency matrix

    Args:
        Adj (array): Input adjacency matrix
        
    Returns:
        jsonData (str): Generated jsonData
    '''
    # Generate jsonData
    AdjData = Adj.tolist()
    for i in range(len(AdjData)):
        for j in range(len(AdjData[i])):
            if AdjData[i][j] == np.inf:
                AdjData[i][j] = "inf"
    jsonData = {"adjacency_matrix": AdjData}
    jsonData = json.dumps(jsonData, indent=4)

    return jsonData

def generate_adjacency_matrix_from_json_data(jsonData) -> np.ndarray:
    '''
    Generates a adjacency matrix from jsonData

    Args:
        jsonData (str): Input jsonData

    Returns:
        Adj (array): Generated adjacency matrix
    '''
    # Generate adjacency matrix
    AdjData = jsonData["adjacency_matrix"]
    for i in range(len(AdjData)):
        for j in range(len(AdjData[i])):
            if AdjData[i][j] == "inf":
                AdjData[i][j] = np.inf
    Adj = np.array(AdjData)

    return Adj