"""Graph generation utilities for Proposed_Method system."""

import networkx as nx
import numpy as np


class GraphGenerator:
    """Generate synthetic graphs for testing and evaluation."""
    
    @staticmethod
    def generate_random_graph(n_nodes, edge_probability=0.2, seed=None):
        """Generate Erdős-Rényi random graph.
        
        Args:
            n_nodes (int): Number of nodes
            edge_probability (float): Probability of edge creation
            seed (int): Random seed for reproducibility
            
        Returns:
            networkx.Graph: Generated random graph
        """
        if seed is not None:
            np.random.seed(seed)
        
        return nx.erdos_renyi_graph(n_nodes, edge_probability)
    
    @staticmethod
    def generate_small_world(n_nodes, k=4, p=0.1, seed=None):
        """Generate Watts-Strogatz small-world graph.
        
        Args:
            n_nodes (int): Number of nodes
            k (int): Each node is connected to k nearest neighbors
            p (float): Probability of rewiring each edge
            seed (int): Random seed for reproducibility
            
        Returns:
            networkx.Graph: Generated small-world graph
        """
        if seed is not None:
            np.random.seed(seed)
            
        return nx.watts_strogatz_graph(n_nodes, k, p)
    
    @staticmethod
    def generate_scale_free(n_nodes, alpha=0.41, beta=0.54, gamma=0.05, seed=None):
        """Generate Barabási-Albert scale-free graph.
        
        Args:
            n_nodes (int): Number of nodes
            alpha (float): Probability for adding a new edge
            beta (float): Probability for edge rewiring
            gamma (float): Probability for edge addition
            seed (int): Random seed for reproducibility
            
        Returns:
            networkx.Graph: Generated scale-free graph
        """
        if seed is not None:
            np.random.seed(seed)
            
        return nx.powerlaw_cluster_graph(n_nodes, int(n_nodes * 0.1), 0.5)
    
    @staticmethod
    def generate_community_graph(n_communities=3, nodes_per_community=10, 
                               p_intra=0.7, p_inter=0.01, seed=None):
        """Generate graph with community structure.
        
        Args:
            n_communities (int): Number of communities
            nodes_per_community (int): Nodes in each community
            p_intra (float): Probability of intra-community edges
            p_inter (float): Probability of inter-community edges
            seed (int): Random seed for reproducibility
            
        Returns:
            networkx.Graph: Generated community graph
        """
        if seed is not None:
            np.random.seed(seed)
            
        n_nodes = n_communities * nodes_per_community
        g = nx.Graph()
        g.add_nodes_from(range(n_nodes))
        
        # Add intra-community edges
        for c in range(n_communities):
            nodes = range(c * nodes_per_community, (c + 1) * nodes_per_community)
            for i in nodes:
                for j in nodes:
                    if i < j and np.random.random() < p_intra:
                        g.add_edge(i, j)
        
        # Add inter-community edges
        for c1 in range(n_communities):
            nodes1 = range(c1 * nodes_per_community, (c1 + 1) * nodes_per_community)
            for c2 in range(c1 + 1, n_communities):
                nodes2 = range(c2 * nodes_per_community, (c2 + 1) * nodes_per_community)
                for i in nodes1:
                    for j in nodes2:
                        if np.random.random() < p_inter:
                            g.add_edge(i, j)
        
        return g
    
    @staticmethod
    def add_node_features(graph, feature_dim=10, feature_type='random', seed=None):
        """Add synthetic node features to graph.
        
        Args:
            graph (networkx.Graph): Input graph
            feature_dim (int): Dimension of node features
            feature_type (str): Type of features ('random', 'degree', 'clustering')
            seed (int): Random seed for reproducibility
            
        Returns:
            networkx.Graph: Graph with node features added
        """
        if seed is not None:
            np.random.seed(seed)
            
        g = graph.copy()
        
        if feature_type == 'random':
            for node in g.nodes():
                g.nodes[node]['features'] = np.random.randn(feature_dim)
                
        elif feature_type == 'degree':
            degrees = dict(g.degree())
            max_degree = max(degrees.values())
            for node in g.nodes():
                base_feature = np.zeros(feature_dim)
                base_feature[0] = degrees[node] / max_degree
                g.nodes[node]['features'] = base_feature
                
        elif feature_type == 'clustering':
            clustering = nx.clustering(g)
            for node in g.nodes():
                base_feature = np.zeros(feature_dim)
                base_feature[0] = clustering[node]
                g.nodes[node]['features'] = base_feature
                
        return g
