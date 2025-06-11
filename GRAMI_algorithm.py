import networkx as nx
from networkx.algorithms import isomorphism
import matplotlib.pyplot as plt


def load_lg(file_path: str) -> nx.Graph:
    """Load a graph in LG format into a NetworkX Graph."""
    G = nx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # skip comments or graph header
            parts = line.split()
            if parts[0] == 'v':
                # Vertex line: v <id> <label>
                vid = int(parts[1])
                vlabel = parts[2]
                try:
                    vlabel = int(vlabel)  # use numeric label if possible
                except:
                    pass
                G.add_node(vid, label=vlabel)
            elif parts[0] == 'e':
                # Edge line: e <src> <dst> <label>
                u = int(parts[1]); v = int(parts[2]); elabel = parts[3]
                try:
                    elabel = int(elabel)
                except:
                    pass
                # For undirected graph, add one edge (NetworkX Graph is undirected by default here)
                G.add_edge(u, v, label=elabel)
    return G

def initial_frequent_edges(G: nx.Graph, tau: int):
    """Identify all frequent single-edge patterns (with MNI support >= tau).
    Returns a list of pattern graphs and a list of edge-type keys (label1, edge_label, label2)."""
    edge_type_info = {}
    # Gather distinct edge types and involved node sets
    for u, v, attr in G.edges(data=True):
        lu = G.nodes[u]['label']; lv = G.nodes[v]['label']; e = attr.get('label')
        # Use an order-independent key for undirected edge (smallest label first)
        if lu <= lv:
            L1, L2 = lu, lv
        else:
            L1, L2 = lv, lu
        key = (L1, e, L2)
        if key not in edge_type_info:
            edge_type_info[key] = {'L1': L1, 'L2': L2, 'edge_label': e,
                                   'nodes_L1': set(), 'nodes_L2': set()}
        # Add the endpoints to respective sets for support counting
        if L1 == L2:
            # If labels are the same, both u and v belong to both sets
            edge_type_info[key]['nodes_L1'].update([u, v])
            edge_type_info[key]['nodes_L2'].update([u, v])
        else:
            if G.nodes[u]['label'] == L1:
                edge_type_info[key]['nodes_L1'].add(u)
                edge_type_info[key]['nodes_L2'].add(v)
            else:
                edge_type_info[key]['nodes_L1'].add(v)
                edge_type_info[key]['nodes_L2'].add(u)
    # Build patterns for each frequent edge type
    frequent_edges = []
    fedge_keys = []  # store keys for extension step
    for key, info in edge_type_info.items():
        L1, L2, e = info['L1'], info['L2'], info['edge_label']
        # MNI support = min(distinct nodes with label L1, distinct nodes with label L2)
        support = min(len(info['nodes_L1']), len(info['nodes_L2']))
        if support >= tau:
            # Create a pattern Graph with two nodes and one edge
            patt = nx.Graph()
            patt.add_node(0, label=L1)
            patt.add_node(1, label=L2)
            patt.add_edge(0, 1, label=e)
            patt.graph['support'] = support  # store support value (optional)
            frequent_edges.append(patt)
            fedge_keys.append(key)
    return frequent_edges, fedge_keys

def is_new_pattern(pattern: nx.Graph, seen_patterns: list) -> bool:
    """Check if a pattern (Graph) is isomorphic to any pattern in seen_patterns."""
    for seen in seen_patterns:
        if (pattern.number_of_nodes() == seen.number_of_nodes() and
            pattern.number_of_edges() == seen.number_of_edges()):
            # Only attempt isomorphism if basic size matches
            iso_matcher = isomorphism.GraphMatcher(pattern, seen,
                        node_match=isomorphism.categorical_node_match('label', None),
                        edge_match=isomorphism.categorical_edge_match('label', None))
            if iso_matcher.is_isomorphic():
                return False
    return True

def is_frequent_pattern(pattern: nx.Graph, G: nx.Graph, tau: int) -> bool:
    """CSP-based frequency check (MNI support) for a given pattern in graph G."""
    # Make a relabeled copy of pattern with node IDs 0..n-1 for convenience
    patt = pattern.copy()
    mapping = {node: i for i, node in enumerate(pattern.nodes())}
    patt = nx.relabel_nodes(patt, mapping, copy=True)
    n = patt.number_of_nodes()
    # Initialize domain of each pattern node: all data nodes with matching label and sufficient degree
    domain = {}
    for i in patt.nodes():
        label_i = patt.nodes[i]['label']
        deg_i = patt.degree(i)
        # Candidates: nodes in G with the same label and degree >= deg_i (basic pruning)
        domain[i] = {v for v, data in G.nodes(data=True)
                     if data['label'] == label_i and G.degree(v) >= deg_i}
    # Enforce arc consistency on domains (AC-3 algorithm)
    pattern_edges = [(u, v, patt.edges[u, v]['label']) for u, v in patt.edges()]
    # Use a queue of constraints to enforce until no change
    constraints = [(u, v, lbl) for (u, v, lbl) in pattern_edges] + [(v, u, lbl) for (u, v, lbl) in pattern_edges]
    # (We add both directions (u->v and v->u) for undirected edge consistency)
    while constraints:
        u, v, lbl = constraints.pop(0)
        to_remove = []
        for a in list(domain[u]):
            # Remove a if no neighbor b in domain[v] satisfies the edge constraint
            valid = False
            for b in domain[v]:
                if G.has_edge(a, b) and G[a][b].get('label') == lbl:
                    valid = True
                    break
            if not valid:
                to_remove.append(a)
        if to_remove:
            for a in to_remove:
                domain[u].remove(a)
            if len(domain[u]) < tau:
                # Domain fell below tau -> cannot reach support τ
                return False
            # If domain changed, add neighbors' constraints back to queue
            for (x, y, el) in pattern_edges:
                if x != v and y == u:
                    constraints.append((x, y, el))
                if x == u and y != v:
                    constraints.append((y, x, el))
    # If any domain has fewer than τ candidates, pattern can't be frequent
    for i in patt.nodes():
        if len(domain[i]) < tau:
            return False
    # Backtracking search for subgraph embeddings, with early stopping when support confirmed
    assigned = {}        # pattern node -> chosen data node mapping
    used_data_nodes = set()  # to ensure injective mapping (no reuse of a data node)
    # Structures to record distinct nodes used for each pattern node across all solutions
    distinct_mappings = {i: set() for i in patt.nodes()}
    found_enough = False

    # Choose the next unassigned pattern node (heuristic: smallest domain first)
    def choose_next_node():
        unassigned = [i for i in patt.nodes() if i not in assigned]
        # pick node with smallest domain (excluding already used nodes)
        return min(unassigned, key=lambda i: len(domain[i] - used_data_nodes))

    def backtrack():
        nonlocal found_enough
        if found_enough:
            return True  # stop if we already found enough support
        if len(assigned) == n:
            # Found a full embedding (a solution mapping for all pattern nodes)
            # Mark the distinct data nodes used for each pattern node
            for i, v in assigned.items():
                distinct_mappings[i].add(v)
            # Check if each pattern node has ≥ τ distinct mappings
            if all(len(distinct_mappings[i]) >= tau for i in patt.nodes()):
                found_enough = True  # pattern is frequent
                return True
            # Continue searching for more solutions (to accumulate distinct mappings)
            return False
        # Choose a pattern node to assign next
        i = choose_next_node()
        for v in list(domain[i] - used_data_nodes):
            # Try assigning pattern node i -> data node v
            assigned[i] = v
            used_data_nodes.add(v)
            # Forward-checking: restrict neighbors' domains based on this assignment
            prune_info = []  # to restore domains after backtracking
            consistent = True
            for (u, w, el) in pattern_edges:
                # If i is part of this edge, enforce neighbor constraint
                if u == i or w == i:
                    j = w if u == i else u  # j is the other end of this edge
                    if j in assigned:
                        # If neighbor j already assigned, check edge consistency
                        if not (G.has_edge(v, assigned[j]) and G[v][assigned[j]].get('label') == el):
                            consistent = False
                            break
                    else:
                        # Filter domain[j] to nodes adjacent to v with the required edge label
                        prune_info.append((j, set(domain[j])))  # save current domain for j
                        domain[j] = {b for b in domain[j] if G.has_edge(v, b) and G[v][b].get('label') == el}
                        # Also ensure no already-used node remains in domain[j]
                        domain[j] -= used_data_nodes
                        if not domain[j]:
                            consistent = False
                            break
            if consistent and backtrack():
                return True
            # Backtrack: restore domains and assignment
            for (j, old_dom) in prune_info:
                domain[j] = old_dom  # restore domain of neighbor j
            used_data_nodes.remove(v)
            del assigned[i]
        return False

    backtrack()
    return found_enough

def grami_mine(G: nx.Graph, tau: int):
    """Main function to mine frequent subgraphs from graph G with support threshold tau.
    Returns a list of frequent patterns (as NetworkX graphs)."""
    frequent_patterns = []
    seen_patterns = []  # to store patterns already generated (to avoid duplicates)

    # 1. Initial frequent edges
    fedge_patterns, fedge_keys = initial_frequent_edges(G, tau)
    for patt in fedge_patterns:
        if is_new_pattern(patt, seen_patterns):
            seen_patterns.append(patt)
            # (Single-edge patterns from initial_frequent_edges are by definition frequent)
            frequent_patterns.append(patt)

    # 2. Recursive subgraph expansion
    def extend_pattern(pattern: nx.Graph):
        """Try all possible one-edge extensions of the given pattern."""
        # Current pattern's node labels for quick reference
        nodes_data = list(pattern.nodes(data=True))
        n_nodes = pattern.number_of_nodes()

        # (a) Attempt to add a new vertex connected by a frequent edge
        for u, data in nodes_data:
            label_u = data['label']
            for (L1, e_label, L2) in fedge_keys:
                # If pattern node u's label matches one end of a frequent edge type, attach new node for the other end
                if label_u == L1:
                    # Add new node with label L2 connected via e_label
                    new_patt = pattern.copy()
                    new_node_id = max(new_patt.nodes()) + 1  # next new node id
                    new_patt.add_node(new_node_id, label=L2)
                    new_patt.add_edge(u, new_node_id, label=e_label)
                    if is_new_pattern(new_patt, seen_patterns):
                        seen_patterns.append(new_patt)
                        if is_frequent_pattern(new_patt, G, tau):
                            frequent_patterns.append(new_patt)
                            extend_pattern(new_patt)  # recursive extension
                if label_u == L2 and L1 != L2:
                    # If u matches the second label, attach new node with the first label (for asymmetric cases)
                    new_patt = pattern.copy()
                    new_node_id = max(new_patt.nodes()) + 1
                    new_patt.add_node(new_node_id, label=L1)
                    new_patt.add_edge(u, new_node_id, label=e_label)
                    if is_new_pattern(new_patt, seen_patterns):
                        seen_patterns.append(new_patt)
                        if is_frequent_pattern(new_patt, G, tau):
                            frequent_patterns.append(new_patt)
                            extend_pattern(new_patt)

        # (b) Attempt to add an edge between existing vertices (closing a cycle)
        # Consider each pair of distinct vertices in the pattern
        n_list = list(pattern.nodes())
        for i in range(len(n_list)):
            for j in range(i+1, len(n_list)):
                u = n_list[i]; w = n_list[j]
                if pattern.has_edge(u, w):
                    continue  # already connected
                label_u = pattern.nodes[u]['label']; label_w = pattern.nodes[w]['label']
                # Check if an edge type between label_u and label_w is frequent
                # We use the same key logic (order labels) to see if this edge type is in fedge_keys
                if label_u <= label_w:
                    maybe_key = (label_u, None, label_w)  # None for edge label placeholder
                else:
                    maybe_key = (label_w, None, label_u)
                # We need to try all possible frequent edge labels between these two labels:
                for (L1, e_label, L2) in fedge_keys:
                    if L1 == maybe_key[0] and L2 == maybe_key[2]:
                        # Found a frequent edge type connecting these labels
                        new_patt = pattern.copy()
                        new_patt.add_edge(u, w, label=e_label)
                        if is_new_pattern(new_patt, seen_patterns):
                            seen_patterns.append(new_patt)
                            if is_frequent_pattern(new_patt, G, tau):
                                frequent_patterns.append(new_patt)
                                extend_pattern(new_patt)
    # Initiate recursion for each initial pattern
    for patt in list(frequent_patterns):
        extend_pattern(patt)

    return frequent_patterns

# --- Example usage (for testing or demonstration) ---
if __name__ == "__main__":
    # Load an example graph (LG format file)
    G = load_lg("datasets/citeseer.lg")  # user-provided LG graph file
    min_support = 2
    patterns = grami_mine(G, min_support)
    print(f"Found {len(patterns)} frequent subgraph patterns.")
    # Visualize patterns or output them (for example, print each pattern's edges and labels)
    for idx, patt in enumerate(patterns, 1):
        nodes = [(n, patt.nodes[n]['label']) for n in patt.nodes()]
        edges = [(u, v, patt.edges[u, v]['label']) for u, v in patt.edges()]
        print(f"Pattern {idx}: Nodes {nodes}, Edges {edges}")

    # Visualize the last frequent pattern
    if patterns:
        patt = patterns[-1]  # Show the last pattern
        pos = nx.spring_layout(patt, seed=42)
        nx.draw_networkx_nodes(patt, pos, node_color='lightblue', node_size=500)
        labels = {n: patt.nodes[n]['label'] for n in patt.nodes()}
        nx.draw_networkx_labels(patt, pos, labels=labels)
        nx.draw_networkx_edges(patt, pos)
        edge_labels = {(u, v): patt.edges[u, v]['label'] for u, v in patt.edges()}
        nx.draw_networkx_edge_labels(patt, pos, edge_labels=edge_labels)
        plt.title("Example Frequent Pattern")
        plt.savefig("frequent_pattern_test2.png")
        plt.show()