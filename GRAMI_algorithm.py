import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


class Graph:
    def __init__(self):
        self.G = nx.Graph()

    def add_edge(self, u, v, label_u=None, label_v=None):
        self.G.add_edge(u, v)
        if label_u: self.G.nodes[u]['label'] = label_u
        if label_v: self.G.nodes[v]['label'] = label_v

    def visualize(self, title="Input Graph"):
        labels = nx.get_node_attributes(self.G, 'label')
        pos = nx.spring_layout(self.G)
        plt.figure(figsize=(6, 4))
        nx.draw(self.G, pos, with_labels=True, node_color='skyblue', node_size=1000)
        nx.draw_networkx_labels(self.G, pos, labels)
        plt.title(title)
        plt.show()


class GRAMI:
    def __init__(self, graph, min_support=2):
        self.graph = graph.G
        self.min_support = min_support
        self.frequent_patterns = []

    def mni_support(self, nodes):
        support_nodes = set()
        for node in self.graph.nodes():
            if all(neigh in self.graph[node] for neigh in nodes if neigh != node):
                support_nodes.add(node)
        return len(support_nodes)

    def mine_frequent_edges(self):
        for u, v in self.graph.edges():
            pattern = frozenset([u, v])
            support = self.mni_support(pattern)
            if support >= self.min_support:
                self.frequent_patterns.append((pattern, support))

    def visualize_patterns(self):
        for i, (pattern, support) in enumerate(self.frequent_patterns):
            subg = self.graph.subgraph(pattern)
            pos = nx.spring_layout(subg)
            plt.figure(figsize=(4, 3))
            nx.draw(subg, pos, with_labels=True, node_color='lightgreen', node_size=800)
            plt.title(f"Pattern {i + 1} | Support: {support}")
            plt.show()


# === MAIN EXECUTION ===

if __name__ == "__main__":
    # Build sample graph
    G = Graph()
    G.add_edge(1, 2, "A", "B")
    G.add_edge(2, 3, "B", "C")
    G.add_edge(3, 4, "C", "A")
    G.add_edge(1, 3, "A", "C")
    G.add_edge(4, 5, "A", "B")

    G.visualize("Original Graph")

    miner = GRAMI(G, min_support=2)
    miner.mine_frequent_edges()
    miner.visualize_patterns()
