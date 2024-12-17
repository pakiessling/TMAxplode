import pytest
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from collections import deque
from tmaxplode.main import explode, construct_adjacency_matrix, find_connected

# Assuming the functions `explode`, `construct_adjacency_matrix`, and `find_connected` are imported from the module.

@pytest.fixture
def sample_data():
    # Sample data for testing (e.g., coordinates of 5 points)
    coords = np.array([
        [0, 0],
        [1, 1],
        [1, 0],
        [0, 1],
        [2, 2]
    ])
    return coords


@pytest.fixture
def sample_ann_data(sample_data):
    import anndata
    # Create an AnnData object with some dummy data
    adata = anndata.AnnData(X=np.random.rand(5, 5))
    adata.obsm["spatial"] = sample_data
    return adata


def test_construct_adjacency_matrix_radius(sample_data):
    # Test the adjacency matrix creation with a radius
    radius = 1.5
    adj_matrix = construct_adjacency_matrix(sample_data, radius)

    # Check if it's a sparse matrix
    assert isinstance(adj_matrix, csr_matrix)
    
    # Check matrix dimensions
    assert adj_matrix.shape == (5, 5)
    
    # Check the connections within radius
    # Points 0, 1, 2, 3 are all within radius of 1.5 (based on simple coordinates)
    assert adj_matrix[0, 1] == 1
    assert adj_matrix[0, 2] == 1
    assert adj_matrix[1, 2] == 1
    assert adj_matrix[3, 0] == 1
    assert adj_matrix[4, 0] == 0  # Points 4 and 0 are not within the radius


def test_find_connected(sample_data):
    # Construct adjacency matrix for the sample data
    radius = 1.5
    adj_matrix = construct_adjacency_matrix(sample_data, radius)

    # Find connected components
    component_ids = find_connected(adj_matrix)

    # Check that the connected components are correct
    assert len(component_ids) == 5
    assert len(set(component_ids)) <= 5  # There should be fewer or equal than 5 unique components

    # Test with the min_cells argument
    component_ids_min_cells = find_connected(adj_matrix, min_size=2)
    assert all(id >= 0 for id in component_ids_min_cells)


def test_explode_with_numpy(sample_data):
    # Test the `explode` function with numpy ndarray input
    radius = 1.5
    result = explode(sample_data, radius=radius)

    # Check that the result is a list of component IDs
    assert isinstance(result, list)
    assert len(result) == len(sample_data)
    assert all(isinstance(x, int) for x in result)


def test_explode_with_ann_data(sample_ann_data):
    # Test the `explode` function with AnnData input
    radius = 1.5
    result = explode(sample_ann_data, radius=radius,connectivity_key=None)

    # Check that the result is an AnnData object
    assert isinstance(result, type(sample_ann_data))
    
    # Check that the new column is added to the AnnData object
    assert "separated" in result.obs
    assert result.obs["separated"].dtype.name == "category"
    assert len(result.obs["separated"]) == len(sample_ann_data)


def test_invalid_connectivity_key_in_ann_data(sample_ann_data):
    # Test case where the connectivity key is not found in AnnData object
    with pytest.raises(KeyError):
        explode(sample_ann_data, connectivity_key="invalid_connectivity")


def test_empty_input_data():
    # Test the `explode` function with an empty numpy array
    empty_data = np.empty((0, 2))
    result = explode(empty_data)
    assert result == []


def test_single_point_input():
    # Test the `explode` function with a single point in numpy array
    single_point_data = np.array([[0, 0]])
    result = explode(single_point_data)
    assert result == [0]  # Only one component for the single point


@pytest.mark.parametrize("radius, expected_adj_matrix_size", [(1.0, (5, 5)), (0.5, (5, 5))])
def test_adj_matrix_size_with_different_radius(sample_data, radius, expected_adj_matrix_size):
    # Test that adjacency matrix size remains constant with different radius values
    adj_matrix = construct_adjacency_matrix(sample_data, radius)
    assert adj_matrix.shape == expected_adj_matrix_size


def test_find_connected_with_min_size():
    # Test with a small radius to ensure that some points are disconnected
    sample_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]])  # 5 points, last one isolated
    adj_matrix = construct_adjacency_matrix(sample_data, radius=1)  # Reduce radius to 0.5
    print(adj_matrix.data)
    component_ids = find_connected(adj_matrix, min_size=2)  # Require minimum size of 2
    
    # Check that at least one component is labeled as -1 (indicating it's too small)
    assert -1 in component_ids  # The last point (index 4) should be isolated and labeled as -1
    
    # Check that the first four points are assigned a valid component ID
    assert len(set(component_ids)) > 1  # There should be more than one component
    assert component_ids[4] == -1  # The isolated point should be labeled as -1


@pytest.mark.parametrize("coords, expected_adj_matrix", [
    (np.array([[0, 0], [1, 1], [2, 2]]), csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
])
def test_specific_adjacency_matrix(coords, expected_adj_matrix):
    # Test the construction of a specific adjacency matrix
    adj_matrix = construct_adjacency_matrix(coords, radius=1.5)
    assert (adj_matrix != expected_adj_matrix).nnz == 0  # Check equality by non-zero elements


