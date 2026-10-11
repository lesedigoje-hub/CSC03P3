import numpy as np
import random
import math

''' Affinity functions for experimenting with different similarity measures between an antigen
    (a user preference profile) and an antibody (a movie genre vector) '''

# dot product affinit
def dot_prod_affinity(antigen: list[float], antibody: list[float]) -> float:
    return np.dot(antigen, antibody)

# Euclidean distance affinity
def euclid_distance_affinity(antigen: list[float], antibody: list[float]) -> float:
    squared_diff = 0

    for x, y in zip(antigen, antibody):
        squared_diff += (x-y)**2

    distance = np.sqrt(squared_diff)

    return 1 / (1 + distance)

# RBF Affinity
def rbf_affinity(antigen: list[float], antibody: list[float], gamma: float = 1.0) -> float:
    squared_diff = 0

    for x, y in zip(antigen, antibody):
        squared_diff += (x - y) ** 2

    return math.exp(-gamma * squared_diff)

# helper function
def norm(alc: list[float]) -> float:
    squared_sum = sum([x**2 for x in alc])

    return np.sqrt(squared_sum)


# Cosine similarity affinity
def cosine_sim_affinity(antigen: list[float], antibody: list[float]) -> float:
    denominator = norm(antigen) * norm(antibody)

    # to avoid division by zero
    if denominator == 0:
        return 0.0

    return np.dot(antigen, antibody) / denominator

# The aiNet parts

# A population of antibodies (movie genre vectors) is repeatedly evaluated against one fixed antigen (user preference profile)
def evaluate_population(population: list[list[float]], antigen: list[float], affinity_function=cosine_sim_affinity) -> list[tuple[list[float], float]]:
    affinities = []

    for antibody in population:
        affinities.append((antibody, affinity_function(antibody, antigen)))

    return affinities

# select a subset of n_best antibodies in the population with the highest affinities to an encountered antigen
def select_best(affinities: list[tuple[list[float], float]], n_best: int) -> list[tuple[list[float], float]]:
    best_n = []
    c_affinities = affinities.copy()

    # To avoid errors if the n_best parameter is greater than the number of available antibodies in the population
    n_best = min(n_best, len(c_affinities))

    while len(best_n) < n_best:
        best = c_affinities[0]
        best_idx = 0

        for i in range(1, len(c_affinities)):
            if c_affinities[i][1] > best[1]:
                best = c_affinities[i]
                best_idx = i

        best_n.append(best)
        del c_affinities[best_idx]

    return best_n

# make n_clones copies of an antibody
def clone(antibody: list[float], n_clones: int) -> list[list[float]]:
    return [antibody.copy() for i in range(n_clones)]

# apply mutation to an antibody according to a given mutation rate and apply clamping to each of the values in range [0,1]
def mutate_antibody(antibody: list[float], mutation_rate: float, mutation_bounds: tuple[float, float] = (-0.1, 0.1)) -> list[float]:
    mutated = antibody.copy()
    lower_bound, upper_bound = mutation_bounds

    for i in range(len(mutated)):
        r = random.random()

        if r < mutation_rate:
            mutated[i] += random.uniform(lower_bound, upper_bound)

            if mutated[i] < 0:
                mutated[i] = 0

            if mutated[i] > 1:
                mutated[i] = 1

    # rounding for easier reading
    # mutated = [round(x, 3) for x in mutated]

    return mutated

# more affinity antibodies get less mutation and low affinity ones get more mutation
def hypermutation(antibody: list[float], affinity: float, mutation_bounds: tuple[float, float] = (-0.1, 0.1)) -> list[float]:
    mutation_rate = 1 - affinity

    return mutate_antibody(antibody, mutation_rate, mutation_bounds)

def clone_and_hypermutate(selected: list[tuple[list[float], float]], n_clones: int,
                           mutation_bounds: tuple[float, float] = (-0.1, 0.1)) -> list[list[float]]:
    hypermutations = []

    for antibody in selected:
        copies = clone(antibody[0], n_clones)

        for copy in copies:
            hypermutations.append(hypermutation(copy, antibody[1], mutation_bounds))

    return hypermutations

# combine population evaluations with hypermutation evaluations and select the best len(population) candidates
def replace_population(population: list[list[float]], hypermutation_affinities: list[tuple[list[float], float]],
                        antigen: list[float], affinity_function=cosine_sim_affinity) -> list[list[float]]:
    population_size = len(population)

    population_affinities = evaluate_population(population, antigen, affinity_function)

    candidates_pool = population_affinities + hypermutation_affinities

    # get a new population
    new_population = select_best(candidates_pool, population_size)

    return [x[0] for x in new_population]

# calculating the euclidean distance between two antibodies
def calculate_distance(antibody_1: list[float], antibody_2: list[float]) -> float:
    squared_sum = 0

    for x, y in zip(antibody_1, antibody_2):
        squared_diff = (x-y) ** 2
        squared_sum += squared_diff

    return math.sqrt(squared_sum)


def suppress(population_affinities: list[tuple[list[float], float]], threshold: float) -> list[list[float]]:

    # using a set to avoid duplicates
    suppressed = set()

    for i in range(len(population_affinities)):
        if i in suppressed:
            continue

        for j in range(i + 1, len(population_affinities)):
            if j in suppressed:
                continue

            antibody_i, affinity_i = population_affinities[i]
            antibody_j, affinity_j = population_affinities[j]

            distance = calculate_distance(antibody_i, antibody_j)

            # if the antibodies are too similar
            if distance < threshold:

                # suppress the one with a lower affinity to the encountered antigen
                if affinity_i >= affinity_j:
                    suppressed.add(j)
                else:
                    suppressed.add(i)
                    break

    diverse_population = []

    # get antibodies corresponding to the indices that were not suppressed
    for i in range(len(population_affinities)):
        if i not in suppressed:
            diverse_population.append(population_affinities[i][0])

    return diverse_population

''' each antibody's net stimulation is its affinity to the antigen
    minus the mean of its affinities to all other antibodies in the population.
    Antibodies whose net stimulation falls below the stimulation_threshold are removed. '''

def compute_stimulation(population_affinities: list[tuple[list[float], float]],
                        affinity_function) -> list[tuple[list[float], float, float]]:
    antibodies = [antibody for antibody, affinity in population_affinities]
    n = len(antibodies)
    stimulations = []

    for i, (antibody_i, antigen_affinity_i) in enumerate(population_affinities):

        if n > 1:
            # mean affinity of this antibody to all other antibodies in the population
            suppression_signal = sum(affinity_function(antibody_i, antibodies[j]) for j in range(n) if j != i) / (n - 1)
        else:
            suppression_signal = 0.0

        net_stimulation = antigen_affinity_i - suppression_signal
        stimulations.append((antibody_i, antigen_affinity_i, net_stimulation))

    return stimulations


def stimulate(stimulations: list[tuple[list[float], float, float]],
              stimulation_threshold: float) -> list[list[float]]:
    # keep only antibodies whose net stimulation meets or exceeds the threshold
    return [antibody for antibody, affinity, net_stimulation in stimulations
            if net_stimulation >= stimulation_threshold]

''' create a random antibody. To be used for replenishing the population after suppression/stimulation
    and to increase diversity '''

def random_antibody(size: int) -> list[float]:
    antibody = []

    for i in range(size):
        antibody.append(random.randint(0, 1))

    return antibody

# add random antibodies to the population until a target population size is reached
def replenish_population(population: list[list[float]], target_size: int, antibody_size: int) -> list[list[float]]:
    new_population = population.copy()

    while (len(new_population) < target_size):
        new_population.append(random_antibody(antibody_size))

    return new_population

def ainet_iteration(population: list[list[float]], antigen: list[float], n_best: int, n_clones: int,
                     suppression_threshold: float, stimulation_threshold: float,
                     affinity_function=euclid_distance_affinity,
                     mutation_bounds: tuple[float, float] = (-0.1, 0.1)) -> list[list[float]]:
    population_affinities = evaluate_population(population, antigen, affinity_function)
    selected_best = select_best(population_affinities, n_best)
    hypermutations = clone_and_hypermutate(selected_best, n_clones, mutation_bounds)
    hypermutations_affinities = evaluate_population(hypermutations, antigen, affinity_function)
    new_population = replace_population(population, hypermutations_affinities, antigen, affinity_function)
    suppressed = suppress(evaluate_population(new_population, antigen, affinity_function), suppression_threshold)
    stimulations = compute_stimulation(evaluate_population(suppressed, antigen, affinity_function), affinity_function)
    stimulated = stimulate(stimulations, stimulation_threshold)
    replenished_population = replenish_population(stimulated, len(population), len(antigen))

    return replenished_population
