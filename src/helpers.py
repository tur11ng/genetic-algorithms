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


def get_nearest_neighbors(individuals: np.ndarray, individual: int, n_neighbors: int,
                          metric=lambda x, y: (pearsonr(x, y)[0]+1)/2) -> np.ndarray:
    individuals = np.copy(individuals)
    individual = np.copy(individuals[individual])
    distances = []
    for i in range(individuals.shape[0]):
        distances.append(metric(individual, individuals[i]))

    sorted_distances_ind = np.argsort(np.array(distances))[-11:-1]
    sorted_distances_ind = sorted_distances_ind[::-1]
    return individuals[sorted_distances_ind]


def repair_individuals(individuals: list, user: list):
    for individual in individuals:
        for i in range(len(individual)):
            if not math.isnan(user[i]):
                individual[i] = user[i]


def replace_invalid_individuals_with_random(individuals: list, user: list):
    for individual_ind in range(len(individuals)):
        if is_individual_valid(individuals[individual_ind], user):
            for replacer_ind in np.random.permutation(len(individuals)):
                if is_individual_valid(individuals[replacer_ind], user):
                    individuals[individual_ind] = individuals[replacer_ind]


def replace_invalid_individuals_with_elite(individuals: list, user: list, elite: list):
    for individual_ind in range(len(individuals)):
        if is_individual_valid(individuals[individual_ind], user):
            individuals[individual_ind] = elite


def is_individual_valid(individual: list, user: list):
    for i, j in zip(individual, user):
        if not math.isnan(j) and i != j:
            return False
    return True


def show_plot(plots, x_label: str, y_label: str, title: str, legend: list, params: dict, save=True):
    fig, ax = plt.subplots()

    for plot in plots:
        plt.plot([point[0] for point in plot], [point[1] for point in plot])
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    textstr = (
        r'POPULATION_SIZE=%d' % (params['POPULATION_SIZE'],),
        r'MAX_GENERATIONS=%d' % (params['MAX_GENERATIONS'],),
        r'P_CROSSOVER=%.3f' % (params['P_CROSSOVER'],),
        r'P_MUTATION=%.3f' % (params['P_MUTATION'],),
    )

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ax.text(0.95, 0.05, '\n'.join(textstr), transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.title("{}".format(title))
    plt.legend(legend)
    plt.show()
    if save:
        fig.savefig("./images/{}.png".format(''.join(textstr)), bbox_inches='tight')
    plt.close()


# Evaluation function with default metric set to normalized pearson correlation (ρ)
def evaluation_function(individual, neighbors, metric=lambda x, y: (pearsonr(x, y)[0] + 1) / 2):
    distances_from_every_neighbor = []
    for neighbor in neighbors:
        distances_from_every_neighbor.append(metric(neighbor, individual))
    avg = np.mean(distances_from_every_neighbor)
    return avg,
