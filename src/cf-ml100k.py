import random
from functools import partial

from deap import base, creator
from deap import tools
from sklearn.metrics import mean_squared_error

from src.helpers import *
from src.parser import *

toolbox = base.Toolbox()


def cf(users, params):
    users_mean_filled = replace_nan_with_mean_columns(users.transpose()).transpose()
    nearest_neighbors = get_nearest_neighbors(users_mean_filled,
                                              params['USER_N']
                                              , 10)
    user = users[params['USER_N']].tolist()

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    generate_individual = partial(replace_nan_with_random, user)
    toolbox.register("individual", lambda ind, ind_gen: ind(ind_gen().tolist()), creator.Individual,
                     generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    evaluate_individual = partial(evaluation_function, neighbors=nearest_neighbors)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutUniformInt, low=1, up=5, indpb=0.5)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=params['POPULATION_SIZE'])

    print("Start of evolution")

    # Evaluate the entire population
    fits = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fits):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    fits = [ind.fitness.values[0] for ind in pop]
    g = 0

    max_history = [(g, max(fits))]
    mean_history = [(g, np.mean(fits))]

    while max(fits) < params['MAX_FIT'] and g < params['MAX_GENERATIONS']:
        g += 1

        print("-- Generation %i --" % g)

        elite_prev = tools.selBest(pop, 1)[0]
        max_prev = max(fits)

        offspring = toolbox.select(pop, len(pop))  # TODO -1?
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < params['P_CROSSOVER']:
                toolbox.mate(child1, child2)

                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < params['P_MUTATION']:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        repair_individuals(offspring, user)
        # replace_invalid_individuals_with_random(offspring, user)

        fits = list(map(toolbox.evaluate, offspring))
        for child, fit in zip(offspring, fits):
            child.fitness.values = fit

        pop[:] = offspring
        if params['ELITISM']:
            pop.append(elite_prev)

        fits = [individual.fitness.values[0] for individual in pop]

        max_history.append((g, max(fits)))
        mean_history.append((g, np.mean(fits)))

        elite = tools.selBest(pop, 1)[0]

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % np.mean(fits))
        print("  Best individual is %s, %s" % (elite, elite.fitness.values))

        # if not max(fits) > max_prev * 1.01:
        #     break

    print("-- End of (successful) evolution --")

    elite = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (elite, elite.fitness.values))

    show_plot([max_history, mean_history], 'Generations', 'Fitness', 'Evolution of fitness across generations',
              ('Max', 'Mean'), params,)

    return elite, max_history, mean_history


if __name__ == "__main__":
    users = parse_ml100k('u.data').to_numpy()

    params = {'USER_N': 89,
              'POPULATION_SIZE': 300,
              'P_CROSSOVER': 0.7,
              'P_MUTATION': 0.3,
              'MAX_GENERATIONS': 142,
              'MAX_FIT': 1,
              'ELITISM': True}

    maxs = []
    generations = []
    for _ in range(10):
        elite, max_history, mean_history = cf(users, params)
        maxs.append(max_history[-1][1])
        generations.append(len(max_history))
    
    print('Average fitness:' + str(np.mean(maxs)))
    print('Average generations:' + str(np.mean(generations)))

    users_train, users_test = prepare_holdout()

    users_train = users_train.to_numpy()
    users_test = users_test.to_numpy()

    user_train = users_train[params['USER_N']]
    user_test = users_test[params['USER_N']]

    for _ in range(1):
        inds = ~np.isnan(user_test)
        elite, max_history, mean_history = cf(users_train, params)
        print("MAE %.4f" % (mean_squared_error(np.array(elite)[inds], user_test[inds])))

