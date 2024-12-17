from collections import deque
from typing import Union, List, Optional
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.neighbors import NearestNeighbors

try:
    import anndata
    AnnData = anndata.AnnData
except ImportError:
    AnnData = None

InputType = Union[np.ndarray, Optional[AnnData]]

def explode(
    data: InputType,
    res_column: str = "separated",
    radius: int = 500,
    min_cells: int = 1,
    spatial_key: str = "spatial",
    connectivity_key: str =  "spatial_connectivities"
) -> InputType:
    """Process the input data while supporting multiple types."""
    
    if isinstance(data, AnnData):
        print("Processing AnnData object")
        if spatial_key not in data.obsm_keys():
            raise KeyError(f"Key '{spatial_key}' not found in AnnData object obsm.")
        if connectivity_key is not None and connectivity_key in data.obsp.keys():
            print(f"Found '{connectivity_key}' in AnnData object. Using it as adjacency matrix.")
            adj = data.obsp[connectivity_key]
        elif connectivity_key is not None and connectivity_key not in data.obsp.keys():
            raise KeyError(f"Key '{connectivity_key}' not found in AnnData object obsp.")
        else:
            print(f"Did not find '{connectivity_key}' in AnnData object.")
            adj = construct_adjacency_matrix(data.obsm[spatial_key], radius=radius)

    else:
        print("Processing input")
        adj = construct_adjacency_matrix(data, radius=radius)

    component_ids = find_connected(adj, min_cells)

    if isinstance(data, AnnData):
        data.obs[res_column] = component_ids
        data.obs[res_column] = data.obs[res_column].astype("category")
        return data
    return component_ids


def construct_adjacency_matrix(coords, radius, return_distance=False, set_diag=False):
    """
    Construct an adjacency matrix using NearestNeighbors for radius-based neighbor search and sparse matrix representation.

    Args:
        coords (np.ndarray): A 2D array of spatial coordinates.
        radius (float): The maximum radius to search for neighbors.

    Returns:
        csr_matrix: The adjacency matrix in sparse CSR format.
        csr_matrix (optional): The distance matrix in sparse CSR format (if return_distance=True).
    """
    print("Constructing adjacency matrix")

    if coords.shape[0] == 0:
        return csr_matrix((0, 0))

    N = coords.shape[0]

    # Use NearestNeighbors for efficient radius-based neighbor search
    tree = NearestNeighbors(radius=radius, metric='euclidean')
    tree.fit(coords)

    # Perform radius search
    distances, indices = tree.radius_neighbors(coords)

    # Prepare data for sparse matrix construction
    row_indices = []
    col_indices = []
    data = []

    for i, (neigh_indices, neigh_distances) in enumerate(zip(indices, distances)):
        for j, neighbor in enumerate(neigh_indices):
            if i < neighbor:  # Avoid duplication of edges (ensure symmetry)
                row_indices.append(i)
                col_indices.append(neighbor)
                data.append(1)
                row_indices.append(neighbor)
                col_indices.append(i)
                data.append(1)
                if return_distance:
                    dist_data.append(neigh_distances[j])
                    dist_data.append(neigh_distances[j])

    # Construct the adjacency matrix
    Adj = csr_matrix((data, (row_indices, col_indices)), shape=(N, N))

    if return_distance:
        dist_data = np.array(dist_data)
        Dst = csr_matrix((dist_data, (row_indices, col_indices)), shape=(N, N))
        return Adj, Dst

    return Adj



def find_connected(connectivity_matrix: csr_matrix, min_size: int = 1) -> List[int]:
    """
    Finds connected components in a graph represented by a sparse binary connectivity matrix.

    Args:
        connectivity_matrix (csr_matrix): Sparse binary adjacency matrix of shape (N, N),
                                          where 1 indicates a connection and 0 indicates no connection.
        min_size (int): Minimum size for a component to be assigned a valid label. Smaller components are labeled -1.

    Returns:
        List[int]: A list where each index represents a node and the value is the component ID.
    """
    num_nodes = connectivity_matrix.shape[0]
    graph = {}

    # Build adjacency list from sparse matrix
    rows, cols = connectivity_matrix.nonzero()
    for u, v in zip(rows, cols):
        if u != v:  # Skip self-loops
            graph.setdefault(u, []).append(v)
            graph.setdefault(v, []).append(u)

    visited = set()
    component_ids = [-1] * num_nodes
    current_component_id = 0

    def flood_fill(start: int) -> List[int]:
        """Performs a BFS to collect all nodes in the same connected component."""
        queue = deque([start])
        visited.add(start)
        component = [start]

        while queue:
            node = queue.popleft()
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    component.append(neighbor)
                    queue.append(neighbor)

        return component

    # Traverse nodes
    for node in range(num_nodes):
        if node not in visited:
            component = flood_fill(node)
            size = len(component)
            if size >= min_size:
                for n in component:
                    component_ids[n] = current_component_id
                current_component_id += 1
            else:
                for n in component:
                    component_ids[n] = -1

    return component_ids
