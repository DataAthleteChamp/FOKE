import networkx as nx
import matplotlib.pyplot as plt
import re

def load_lg(filepath):
    G = nx.Graph()
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                node_id = int(parts[1])
                label = parts[2]
                G.add_node(node_id, label=label)
            elif parts[0] == 'e':
                source = int(parts[1])
                target = int(parts[2])
                label = parts[3]
                G.add_edge(source, target, label=label)
    return G


def visualize_graph(G, title, max_nodes=100):
    if len(G.nodes) > max_nodes:
        print(f"Skipping visualization for '{title}' (too large: {len(G.nodes)} nodes).")
        return

    pos = nx.kamada_kawai_layout(G)
    node_labels = nx.get_node_attributes(G, 'label')
    edge_labels = nx.get_edge_attributes(G, 'label')

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=500)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()


    safe_title = re.sub(r'[^\w\-_\. ]', '_', title)
    filename = f"{safe_title}.png"
    plt.savefig(filename)
    print(f" Saved: {filename}")

    plt.show()



# Load and visualize
datasets = {
    "citeseer.lg": "datasets/citeseer.lg",
    "mico.lg": "datasets/mico.lg",
    "test1.lg": "datasets/test1.lg",
    "test2.lg": "datasets/test2.lg"
}


graphs = {name: load_lg(path) for name, path in datasets.items()}

for name, graph in graphs.items():
     visualize_graph(graph, f"Graph Visualization: {name}")
