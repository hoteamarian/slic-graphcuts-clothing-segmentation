import numpy as np
import networkx as nx
import math


class GraphBuilder:
    """
    Builds a graph using superpixels as nodes.
    Each node corresponds to a superpixel with its color and spatial features,
    and an edge is added between nodes if the corresponding superpixels are adjacent
    in the image. Edge weights are determined by the color similarity.

    Parameters:
    -----------
    beta : float
        Controls the influence of color differences in the edge weights.
    gamma : float
        Scaling factor for the edge weights.
    """

    def __init__(self, beta: float = 0.5, gamma: float = 1.0):
        self.beta = beta
        self.gamma = gamma

    def build_graph(self, label_map: np.ndarray, centers: np.ndarray) -> nx.Graph:
        """
        Builds an undirected graph where each node is a superpixel and an edge
        exists between nodes if the corresponding superpixels are adjacent in the image.

        Parameters:
        -----------
        label_map : np.ndarray
            2D array (of shape [height, width]) containing the superpixel label for each pixel.
        centers : np.ndarray
            Array of superpixel centers and features with shape (num_superpixels, 5).
            The first three elements are the Lab color (L, A, B) and the last two are the spatial coordinates (x, y).

        Returns:
        --------
        G : networkx.Graph
            The constructed graph with nodes and weighted edges.
        """
        height, width = label_map.shape
        G = nx.Graph()

        num_superpixels = centers.shape[0]

        # Add nodes with their features (color and position)
        for i in range(num_superpixels):
            # Unpack features: Lab color and spatial coordinates
            L, A, B, x, y = centers[i]
            G.add_node(i, color=(L, A, B), position=(x, y))

        # Set to store added edges (to avoid duplicates)
        added_edges = set()

        # Iterate over the label map to find adjacent superpixels
        for i in range(height):
            for j in range(width):
                current_label = label_map[i, j]
                if current_label == -1:
                    continue
                # Check right neighbor
                if j + 1 < width:
                    neighbor_label = label_map[i, j + 1]
                    if neighbor_label == -1:
                        continue
                    if neighbor_label != current_label:
                        edge = tuple(sorted((int(current_label), int(neighbor_label))))
                        added_edges.add(edge)
                # Check bottom neighbor
                if i + 1 < height:
                    neighbor_label = label_map[i + 1, j]
                    if neighbor_label == -1:
                        continue
                    if neighbor_label != current_label:
                        edge = tuple(sorted((int(current_label), int(neighbor_label))))
                        added_edges.add(edge)

        # Add edges to the graph with computed weights
        for (i, j) in added_edges:
            # Retrieve the Lab color for both nodes
            color_i = np.array(G.nodes[i]['color'])
            color_j = np.array(G.nodes[j]['color'])
            # Compute Euclidean distance in Lab color space
            color_diff = np.linalg.norm(color_i - color_j)
            # Weight is based on the similarity: high similarity yields high weight
            weight = self.gamma * math.exp(-self.beta * (color_diff ** 2))
            G.add_edge(i, j, weight=weight)

        return G