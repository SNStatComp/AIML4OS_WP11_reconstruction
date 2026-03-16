import ibis

def sample_edges(edges: ibis.Table, num_samples: int) -> ibis.Table:
    """
    Sample a subset of edges from the network.

    Parameters:
    - edges: An Ibis Table representing the edges of the network.
    - num_samples: The number of edges to sample.

    Returns:
    - sampled_edges: An Ibis Table of sampled edges.
    """
    import random

    if num_samples >= len(edges ):
        return edges  # Return all edges if num_samples exceeds available edges

    sampled_edges = random.sample(edges, num_samples)
    return sampled_edges    