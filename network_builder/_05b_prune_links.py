def prune_linkss(network, degree_constraints):
    """
    Prune links from the network based on degree constraints.

    Parameters:
    - network: A list of edges (tuples) representing the network.
    - degree_constraints: A dictionary mapping node IDs to their maximum allowed degree.

    Returns:
    - pruned_network: A list of edges that satisfy the degree constraints.
    """
    from collections import defaultdict

    # Count the degree of each node
    degree_count = defaultdict(int)
    for edge in network:
        node_a, node_b = edge
        degree_count[node_a] += 1
        degree_count[node_b] += 1

    # Prune edges that violate degree constraints
    pruned_network = []
    for edge in network:
        node_a, node_b = edge
        if (degree_count[node_a] <= degree_constraints.get(node_a, float('inf')) and
            degree_count[node_b] <= degree_constraints.get(node_b, float('inf'))):
            pruned_network.append(edge)

    return pruned_network