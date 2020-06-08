import random
from functools import partial

from deap import base, creator
from deap import tools
from scipy.stats import pearsonr

from src.helpers import *
from src.parser import *

random.seed(64)
toolbox = base.Toolbox()

USER_N = 5
POPULATION_SIZE = 20
P_MUTATION = 0.9
P_CROSSOVER = 0.05
MAX_GENERATIONS = 50


# evaluation function based on pearson correlation
def evaluation_function_pearson(individual, neighbors):
    distances_from_every_neighbor = []
    for neighbor in neighbors:
        # distances_from_every_neighbor.append(cosine(neighbor, individual)+1)
        # distances_from_every_neighbor.append(cdist(neighbor, individual, metric='cityblock'))  # Manhattan distance
        distances_from_every_neighbor.append(abs(pearsonr(neighbor, individual)[0]))  # Pearson distance returns (
        # correlation coefficient, p-value)
        avg = np.mean(distances_from_every_neighbor)
    return (avg,)


def main():
    users = parse_ml100k()
    users_mean_filled = replace_nan_with_mean_columns(users.transpose()).transpose()
    nearest_neighbors = get_nearest_neighbors(users_mean_filled,
                                              USER_N
                                              , 10)
    user = users[0].tolist()

    # users = users.tolist()
    # users_mean_filled = users_mean_filled.tolist()
    nearest_neighbors = nearest_neighbors.tolist()

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # register individual
    generate_individual = partial(replace_nan_with_random, user)
    toolbox.register("individual", lambda ind, ind_gen: ind(ind_gen().tolist()), creator.Individual,
                     generate_individual)

    # define the population to be a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # register the goal / fitness function
    evaluation_function = partial(evaluation_function_pearson, neighbors=nearest_neighbors)

    toolbox.register("evaluate", evaluation_function)

    # register the crossover operator
    toolbox.register("mate", tools.cxTwoPoint)
    # toolbox.register("mate", tools.cxOnePoint)
    # toolbox.register("mate", tools.cxUniform)
    # toolbox.register("mate", tools.cxOrdered)
    # toolbox.register("mate", tools.cxPartialyMatched)

    # register a mutation operator TODO 5 or 6?
    toolbox.register("mutate", tools.mutUniformInt, low=1, up=5, indpb=P_MUTATION)

    # register selection operator
    toolbox.register("select", tools.selTournament, tournsize=3)
    # toolbox.register("select", tools.selRoulette)

    pop = toolbox.population(n=POPULATION_SIZE)
    print(pop)
    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for individual, fit in zip(pop, fitnesses):
        individual.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    # Begin the evolution
    while max(fits) < 100 and g < MAX_GENERATIONS:
        g += 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # cross two individuals with probability P_CROSSOVER
            if random.random() < P_CROSSOVER:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability P_MUTATION
            if random.random() < P_MUTATION:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))


if __name__ == "__main__":
    main()
