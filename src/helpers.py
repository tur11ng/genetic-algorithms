import numpy as np
from scipy.stats import pearsonr


def replace_nan_with_random(matrix: np.ndarray, lower=1, upper=6) -> np.ndarray:
    matrix = np.copy(matrix)
    nan_mask = np.isnan(matrix)
    matrix[nan_mask] = np.random.randint(lower, upper, size=np.count_nonzero(nan_mask))
    return matrix


def replace_nan_with_mean_columns(matrix: np.ndarray) -> np.ndarray:
    matrix = np.copy(matrix)
    col_mean = np.nanmean(matrix, axis=0)
    inds = np.where(np.isnan(matrix))
    matrix[inds] = np.take(col_mean, inds[1])
    return matrix


# def get_nearest_neighbors(individuals: np.ndarray, individual: int, n_neighbors) -> np.ndarray:
#     individuals = np.copy(individuals)
#     knn = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='kd_tree')
#     knn.fit(individuals)
#     distances, indices = knn.kneighbors(individuals)
#     nearest_neighbors_ind = indices[individual][1:]
#     return individuals[nearest_neighbors_ind, :]


def get_nearest_neighbors(individuals: np.ndarray, individual: int, n_neighbors: int,
                          metric=lambda x, y: abs(pearsonr(x, y)[0])) -> np.ndarray:
    individuals = np.copy(individuals)
    individual = np.copy(individuals[individual])
    distances = []
    for i in range(individuals.shape[0]):
        distances.append(metric(individual, individuals[i]))

    sorted_distances_ind = np.argsort(np.array(distances))[-11:-1]
    return individuals[sorted_distances_ind]


def mask_matrix(matrix: np.ndarray, mask_matrix: np.ndarray):
    matrix = np.copy(matrix)
    mask_matrix = np.copy(mask_matrix)

    nan_mask = np.isnan(mask_matrix)
    not_nan_mask_matrix = mask_matrix[~nan_mask]

    matrix[~nan_mask] = not_nan_mask_matrix
    return matrix
