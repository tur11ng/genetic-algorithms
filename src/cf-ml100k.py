import random
from functools import partial

from deap import base, creator
from deap import tools

from src.helpers import *
from src.parser import *

toolbox = base.Toolbox()

USER_N = 11
POPULATION_SIZE = 300
P_MUTATION = 0.3
P_CROSSOVER = 0.8
MAX_GENERATIONS = 20
MAX_FIT = 1

def main():
    users = parse_ml100k()
    users_mean_filled = replace_nan_with_mean_columns(users.transpose()).transpose()
    nearest_neighbors = get_nearest_neighbors(users_mean_filled,
                                              USER_N
                                              , 10)
    user = users[USER_N].tolist()

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    generate_individual = partial(replace_nan_with_random, user)
    toolbox.register("individual", lambda ind, ind_gen: ind(ind_gen().tolist()), creator.Individual,
                     generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    evaluate_individual = partial(evaluation_function, neighbors=nearest_neighbors)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=1, up=5, indpb=0.5)
    toolbox.register("select", tools.selTournament, tournsize=4)

    pop = toolbox.population(n=POPULATION_SIZE)

    print("Start of evolution")

    # Evaluate the entire population
    fits = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fits):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    fits = [ind.fitness.values[0] for ind in pop]
    g = 0

    elite_history = [(g, max(fits))]
    mean_history = [(g, np.mean(fits))]

    while max(fits) < MAX_FIT and g < MAX_GENERATIONS:
        g += 1

        print("-- Generation %i --" % g)

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CROSSOVER:
                toolbox.mate(child1, child2)

                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < P_MUTATION:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        for child in offspring:
            repair_individual(child, user)

        invalid_individuals = [individual for individual in offspring if not individual.fitness.valid]
        invalid_individuals_fits = list(map(toolbox.evaluate, invalid_individuals))
        for ind, fit in zip(invalid_individuals, invalid_individuals_fits):
            ind.fitness.values = fit

        print("  Reevaluated %i offsprings with invalid fitness" % len(invalid_individuals))

        pop[:] = offspring

        fits = [individual.fitness.values[0] for individual in pop]

        elite_history.append((g, max(fits)))
        mean_history.append((g, np.mean(fits)))

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % np.mean(fits))

    print("-- End of (successful) evolution --")

    elite = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (elite, elite.fitness.values))

    show_plot(mean_history, 'Generations', 'Evaluation', 'Title', '', save=False)


if __name__ == "__main__":
    main()
