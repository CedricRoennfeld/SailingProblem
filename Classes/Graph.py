# Imports
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm  # Import colormap functionality

class Graph:
    def __init__(self, n, edges=None):
        """
        Initialize a weighted undirected graph without loops.
        :param n: Number of nodes
        :param edges: Adjacency Matrix (has to be symmetric)
        """
        self.nodes = n  # Number of nodes

        # Initialize edges with a zero matrix if no edges are provided
        self.edges = np.zeros((n, n)) if edges is None else np.array(edges)
        assert np.array_equal(np.transpose(self.edges), self.edges), "Adjacency matrix has to be symmetrical"

    def add_edge(self, first, second, weight=1):
        """
        Add an edge between two nodes (`first`, `second`) with an optional weight (default is 1).
        Since this is an undirected graph, the edge is symmetrical.

        :param first: first node
        :param second: second node
        :param weight: weight of the edge
        """
        self.edges[first][second] = weight
        self.edges[second][first] = weight  # Ensure symmetry for undirected graphs

    def plot_graph(self, colormap=cm.jet):
        """
        Plot the graph using matplotlib.
        Nodes are placed in a circular layout, and edges are drawn between them with colors based on their weights.

        :param colormap: matplotlib colormap
        """
        # Check if there are any nodes in the graph
        if self.nodes == 0:
            print("Graph is empty")
            return

        # Calculate positions of nodes on a circle for visualization
        # Starting at pi/2 for the first node being at top
        angles = np.linspace(0.5 * np.pi, 2.5 * np.pi, self.nodes, endpoint=False)
        node_pos = np.array([[np.cos(angle), np.sin(angle)] for angle in angles])

        # Set up the plotting area
        plt.figure(figsize=(8, 8))

        # Normalize weights for color mapping (skip diagonal/zeros)
        non_zero_weights = self.edges[self.edges != 0]
        if len(non_zero_weights) > 0:
            min_weight = np.min(non_zero_weights)
            max_weight = np.max(non_zero_weights)
        else:
            min_weight, max_weight = 0, 1  # Avoid division by zero if no edges exist

        norm = plt.Normalize(min_weight, max_weight)

        # Plot edges: iterate through the upper triangle of the adjacency matrix to prevent redundancy
        for (i, j), weight in np.ndenumerate(self.edges):
            if weight != 0 and i < j:  # Only plot edges with a non-zero weight
                # Normalize the weight and map it to a color
                color = colormap(norm(weight))
                # Draw edges between node i and node j with the corresponding color
                plt.plot([node_pos[i][0], node_pos[j][0]], [node_pos[i][1], node_pos[j][1]], color=color, linewidth=2)

        # Plot nodes
        for i in range(self.nodes):
            plt.plot(node_pos[i][0], node_pos[i][1], 'o', markersize=10, color='lightblue', markeredgecolor='black')

        # Set equal aspect ratio and remove axis for better visual appearance
        plt.gca().set_aspect('equal')
        plt.gca().set_axis_off()
        plt.show()