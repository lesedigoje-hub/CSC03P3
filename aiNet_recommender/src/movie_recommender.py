from . import aiNet
import random

# create a initial population of random candidate movie vectors
def build_initial_population(candidate_movies: list[list[int]], population_size: int) -> list[list[int]]:
    population = []

    for i in range(population_size):
        random_movie = random.choice(candidate_movies)

        population.append(random_movie.genres)

    return population

# get candidate movies that best match the hypermutated vectors which may have drifted away from valid binary vectors
def match_population_to_movies(population: list[list[float]], candidate_movies: list, affinity_function) -> list[tuple]:
    ''' keep only the single best matching movie per population member so the same movie doesn't get recommended multiple times.'''

    best_matching_movies = []
    already_picked = []

    for vector in population:
        best_movie = None
        best_affinity = 0

        for movie in candidate_movies:
            affinity = affinity_function(vector, movie.genres)

            if affinity > best_affinity and movie not in already_picked:
                best_movie = movie
                best_affinity = affinity

        if best_movie is not None:
            best_matching_movies.append((best_movie, best_affinity))
            already_picked.append(best_movie)

    return best_matching_movies

# For taking a snaphot of the current state of the network for plotting purposes
def _take_snapshot(population: list[list[float]], label: str) -> dict:
    return {
        'label': label,
        'population': [v[:] for v in population],
    }

# run the aiNet recommendation
def recommend_for_user(user, movies, affinity_function, population_size, n_iterations,
                        n_best, n_clones, suppression_threshold, stimulation_threshold,
                        top_n_movies, mutation_bounds=(-0.1, 0.1), initial_population=None):

    antigen = user.preference_profile

    # only recommend movies the user hasn't already rated
    candidate_movies = [movie for movie in movies if movie.id not in user.rated_movie_ids]

    if initial_population and len(initial_population) > 0:
        # seed with the carried over population from the previous round
        population = list(initial_population)

        # if the carried over population is smaller than population size, fill the rest with random candidates
        while len(population) < population_size:
            random_movie = random.choice(candidate_movies)
            population.append(random_movie.genres)

        # trim to population_size in case it somehow exceeds it
        population = population[:population_size]
    else:
        population = build_initial_population(candidate_movies, population_size)

    # checkpoints as a dict keyed by step number to avoid duplicates when n_iterations is small and multiple fractions land on the same step
    checkpoint_steps = {
        n_iterations // 4: f'25% (Generation {n_iterations // 4})',
        n_iterations // 2: f'50% (Generation {n_iterations // 2})',
        (3 * n_iterations) // 4: f'75% (Generation {(3 * n_iterations) // 4})',
        n_iterations: f'Final (Generation {n_iterations})',
    }

    # snapshot dict keyed by step, so duplicate steps from small n_iterations collapse cleanly
    snapshots = {}

    # initial population before any evolution
    snapshots[0] = _take_snapshot(population, 'Initial (step 0)')

    for i in range(n_iterations):
        population = aiNet.ainet_iteration(population, antigen, n_best, n_clones,
                                            suppression_threshold, stimulation_threshold,
                                            affinity_function, mutation_bounds)

        step = i + 1
        if step in checkpoint_steps and step not in snapshots:
            snapshots[step] = _take_snapshot(population, checkpoint_steps[step])

    snapshots_list = [snapshots[k] for k in sorted(snapshots.keys())]

    scored_movies = match_population_to_movies(population, candidate_movies, affinity_function)

    scored_movies.sort(key=lambda pair: pair[1], reverse=True)

    top_movies = scored_movies[:top_n_movies]

    ''' carry forward the genre vectors of all matched movies and not just top_n,
     so the full evolved state is preserved for the next round '''
    next_population = [movie.genres for movie, _ in scored_movies]

    return top_movies, next_population, snapshots_list