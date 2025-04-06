import numpy as np
import networkx as nx
from sklearn.mixture import GaussianMixture


class GraphCutSegmenter:
    """
    Solves the graph segmentation using the min-cut (max-flow) algorithm with an iterative
    EM-based Gaussian Mixture Model (GMM) update for the data term.

    This class takes as input the SLIC segmentation output (label_map and centers),
    the interactive seed mask, and the graph built on the superpixels, and then
    constructs a directed graph with terminal nodes ("source" and "sink").

    For each superpixel node:
      - If it is marked as foreground (via interactive seeds), it is forced to the foreground.
      - If it is marked as background, it is forced to the background.
      - If it is unmarked, unary costs (data terms) are computed based on its Lab color likelihood
        from the fitted foreground and background GMMs.

    The method iteratively refines the segmentation by updating the GMM parameters based on
    the current segmentation until convergence.

    Parameters:
    -----------
    sigma : float
        Controls the sensitivity of the data term based on color differences.
    inf_capacity : float
        A very large capacity used to force seed constraints.
    """

    def __init__(self, sigma: float = 10.0, inf_capacity: float = 1e9):
        self.sigma = sigma
        self.inf_capacity = inf_capacity

    def segment(self, label_map: np.ndarray, centers: np.ndarray,
                seed_mask: np.ndarray, G: nx.Graph) -> np.ndarray:
        """
        Performs graph cut segmentation using iterative EM-based GMM updates.

        Parameters:
        -----------
        label_map : np.ndarray
            2D array with superpixel labels for each pixel.
        centers : np.ndarray
            Array of shape (num_superpixels, 5) with each row [L, A, B, x, y] for the superpixel.
        seed_mask : np.ndarray
            Grayscale mask with values:
              0 - no seed,
              1 - foreground (clothing) seed,
              2 - background seed.
        G : nx.Graph
            Undirected graph built using superpixels as nodes with color-based edge weights.

        Returns:
        --------
        segmentation : np.ndarray
            Binary segmentation mask (1 for foreground, 0 for background).
        """
        num_nodes = centers.shape[0]

        # Determine seed assignment for each superpixel:
        # 'foreground' if any pixel in the superpixel is marked with 1,
        # 'background' if any pixel is marked with 2,
        # 'unknown' if no seed is present (if both exist, majority wins).
        seed_assignment = {}
        for i in range(num_nodes):
            indices = np.where(label_map == i)
            fg_count = np.sum(seed_mask[indices] == 1)
            bg_count = np.sum(seed_mask[indices] == 2)
            if fg_count > 0 and bg_count == 0:
                seed_assignment[i] = 'foreground'
            elif bg_count > 0 and fg_count == 0:
                seed_assignment[i] = 'background'
            elif fg_count > 0 and bg_count > 0:
                seed_assignment[i] = 'foreground' if fg_count >= bg_count else 'background'
            else:
                seed_assignment[i] = 'unknown'

        # Initialize current segmentation for superpixels:
        # For forced seeds use the seed assignment; for unknown nodes, assign initial guess (here, background = 0).
        current_segmentation = np.zeros(num_nodes, dtype=np.uint8)
        for i in range(num_nodes):
            if seed_assignment[i] == 'foreground':
                current_segmentation[i] = 1
            elif seed_assignment[i] == 'background':
                current_segmentation[i] = 0
            else:
                current_segmentation[i] = 0  # initial guess for 'unknown'

        max_em_iterations = 10
        for iteration in range(max_em_iterations):
            # --- E-Step: Fit GMMs for foreground and background based on current segmentation ---
            fg_colors = []
            bg_colors = []
            for i in range(num_nodes):
                # Use forced seeds and current segmentation for unknown nodes.
                if seed_assignment[i] == 'foreground' or (
                        seed_assignment[i] == 'unknown' and current_segmentation[i] == 1):
                    fg_colors.append(centers[i, :3])
                elif seed_assignment[i] == 'background' or (
                        seed_assignment[i] == 'unknown' and current_segmentation[i] == 0):
                    bg_colors.append(centers[i, :3])
            fg_colors = np.array(fg_colors) if len(fg_colors) > 0 else np.zeros((1, 3))
            bg_colors = np.array(bg_colors) if len(bg_colors) > 0 else np.zeros((1, 3))

            fg_gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)
            bg_gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)
            fg_gmm.fit(fg_colors)
            bg_gmm.fit(bg_colors)

            # --- Compute Data (Unary) Costs for Each Superpixel ---
            data_costs = {}
            for i in range(num_nodes):
                color = centers[i, :3].reshape(1, -1)
                if seed_assignment[i] == 'unknown':
                    # Compute negative log-likelihood from the GMMs.
                    log_likelihood_fg = fg_gmm.score_samples(color)[0]
                    log_likelihood_bg = bg_gmm.score_samples(color)[0]
                    data_fg = -log_likelihood_fg
                    data_bg = -log_likelihood_bg
                elif seed_assignment[i] == 'foreground':
                    data_fg = 0
                    data_bg = self.inf_capacity  # Force foreground
                elif seed_assignment[i] == 'background':
                    data_fg = self.inf_capacity  # Force background
                    data_bg = 0
                data_costs[i] = (data_fg, data_bg)

            # --- Build the Directed Graph D with Updated Terminal (Data) Edges ---
            D = nx.DiGraph()
            D.add_node('source')
            D.add_node('sink')
            for i in range(num_nodes):
                D.add_node(i)
                fg_cost, bg_cost = data_costs[i]
                D.add_edge('source', i, capacity=fg_cost)
                D.add_edge(i, 'sink', capacity=bg_cost)
            # Add pairwise edges from the undirected graph G.
            for (u, v, data) in G.edges(data=True):
                w = data.get('weight', 1.0)
                D.add_edge(u, v, capacity=w)
                D.add_edge(v, u, capacity=w)

            # --- Solve the Min-Cut Problem ---
            cut_value, partition = nx.minimum_cut(D, 'source', 'sink')
            reachable, _ = partition
            new_segmentation = np.zeros(num_nodes, dtype=np.uint8)
            for i in range(num_nodes):
                if i in reachable:
                    new_segmentation[i] = 1
                else:
                    new_segmentation[i] = 0

            # --- Check for Convergence ---
            if np.array_equal(current_segmentation, new_segmentation):
                break
            current_segmentation = new_segmentation.copy()

        # --- Build the Final Pixel-Level Segmentation Mask ---
        segmentation = np.zeros(label_map.shape, dtype=np.uint8)
        for i in range(num_nodes):
            segmentation[label_map == i] = 1 if current_segmentation[i] == 1 else 0

        return segmentation
