import math

import matplotlib.pyplot as plt
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
    sorted_distances_ind = sorted_distances_ind[::-1]
    return individuals[sorted_distances_ind]


# def mask_matrix(matrix: np.ndarray, mask_matrix: np.ndarray):
#     matrix = np.copy(matrix)
#     mask_matrix = np.copy(mask_matrix)
#
#     matrix[~np.isnan(mask_matrix)] = mask_matrix[~np.isnan(mask_matrix)]
#     return matrix


def repair_individual(individual: list, mask: list):
    for i in range(len(individual)):
        if not math.isnan(mask[i]):
            individual[i] = mask[i]
            # del individual.fitness.values


def show_plot(points, x_label, y_label, title, legend, save=True):
    plt.plot([point[0] for point in points], [point[1] for point in points])
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.title("{}".format(title))
    plt.legend(legend)
    plt.show()
    if save:
        plt.savefig("./images/{}.png".format(title), bbox_inches='tight')
    plt.close()


# Evaluation function with default metric set to normalized pearson correlation (ρ)
def evaluation_function(individual, neighbors, metric=lambda x, y: (pearsonr(x, y)[0] + 1) / 2):
    distances_from_every_neighbor = []
    for neighbor in neighbors:
        distances_from_every_neighbor.append(metric(neighbor, individual))
    avg = np.mean(distances_from_every_neighbor)
    return avg,
