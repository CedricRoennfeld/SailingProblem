# Imports
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm  # Import colormap functionality

class Graph:
    def __init__(self, n, edges=None, nodes=None):
        """
        Initialize a weighted undirected graph without loops.
        :param n: Number of nodes
        :param edges: Adjacency Matrix (has to be symmetric)
        """
        self.n = n  # Number of nodes
        self.nodes = list(range(n)) if nodes is None else nodes

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

    def permute_(self, permutation):
        """
        Permutes the nodes inplace, e.g. [2,0,1] places the 2nd node at the place of the 0th
        :param permutation:
        """
        assert sorted(permutation) == list(range(self.n)), "permutations has to be a permutation of the nodes"

        # Compute the inverse of the permutation
        inverse_permutation = [0] * self.n
        for i, p in enumerate(permutation):
            inverse_permutation[p] = i

        # ix_ creates a matrix permutation of the rows and columns
        self.edges = self.edges[np.ix_(inverse_permutation, inverse_permutation)]
        self.nodes = [self.nodes[i] for i in permutation]

    def permute(self, permutation):
        """
        Permutes the nodes, e.g. [2,0,1] places the 2nd node at the place of the 0th
        :param permutation: inverse permutation
        :return: permuted graph
        """
        assert sorted(permutation) == list(range(self.n)), "permutations has to be a permutation of the nodes"

        # Compute the inverse of the permutation
        #inverse_permutation = [0] * self.nodes
        #for i, p in enumerate(permutation):
        #    inverse_permutation[p] = i

        # ix_ creates a matrix permutation of the rows and columns
        return Graph(self.n, self.edges[np.ix_(permutation, permutation)], [self.nodes[i] for i in permutation])

    def __add__(self, other):
        """
        Adds two graphs of same size together by adding there edge weights.
        :param other:
        """
        assert isinstance(other, Graph) and self.n == other.n, "can only add graphs with the same number of nodes"
        return Graph(self.n, self.edges + other.edges)

    def plot_graph(self, title=None, colormap=cm.jet):
        """
        Plot the graph using matplotlib.
        Nodes are placed in a circular layout, and edges are drawn between them with colors based on their weights.

        :param title: plot title
        :param colormap: matplotlib colormap
        """
        # Check if there are any nodes in the graph
        if self.n == 0:
            print("Graph is empty")
            return

        # Calculate positions of nodes on a circle for visualization
        # Starting at pi/2 for the first node being at top
        angles = np.linspace(0.5 * np.pi, 2.5 * np.pi, self.n, endpoint=False)
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
        for i in range(self.n):
            plt.plot(node_pos[i][0], node_pos[i][1], 'o', markersize=10, color='lightblue', markeredgecolor='black')

            # Add labels (node numbers) slightly outside of the circle
            for i, (x, y) in enumerate(node_pos):
                label_x = x * 1.15
                label_y = y * 1.15
                plt.text(label_x, label_y, str(self.nodes[i]), fontsize=12, ha='center', va='center')

        # Set equal aspect ratio and remove axis for better visual appearance
        plt.gca().set_aspect('equal')
        plt.gca().set_axis_off()
        if title is not None:
            plt.title(title)
        plt.show()