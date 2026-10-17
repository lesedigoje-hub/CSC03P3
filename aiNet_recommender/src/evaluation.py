import math
from .models import Movie
from .aiNet import calculate_distance

# find the best matching recommendation for each held out movie, then average these best match scores accros all the held out movies.
def genre_affinity_score(recommendations: list[tuple[Movie, float]], held_out_movies: list[Movie], affinity_function) -> float:

    # if either list of recommendations or held out movies is empty, there's nothing to compare so we can't produce a meaningful score
    if not recommendations or not held_out_movies:
        return 0.0

    # get recommended movies' vectors the affinity value itself isn't needed here so it's just discarded for each pair
    rec_vectors = [movie.genres for movie, affinity in recommendations]

    # to store the sum of all best match scores
    total = 0.0

    # go through every movie the user actually liked but that was hidden from the recommender
    for movie in held_out_movies:
        # list to hold how similar this held out movie is to each recommended movie
        affinities = []

        # compare this one held out movie's genre vector against every recommended movie's vector
        for vector in rec_vectors:
            affinities.append(affinity_function(vector, movie.genres))

        # get the recommendation with the highest affinity to this held out movie
        best_match = max(affinities)

        # add this held out movie's best score to the total
        total += best_match

    # average the best match scores across every held out movie
    average_best_matches = total / len(held_out_movies)

    return round(average_best_matches, 4)

# Average pairwise Euclidean distance between the genre vectors of all recommended movies.
def intra_list_diversity(recommendations: list[tuple[Movie, float]]) -> float:

    # nothing to measure if movies are less than two because no pair can be formed
    if len(recommendations) < 2:
        return 0.0

    # pull out just the genre vectors from each (movie, affinity) pair
    vectors = [movie.genres for movie, affinity in recommendations]

    # to store the total of all the pairwise distances
    total_distance = 0.0

    # to store the number of pairs we compared, used to compute the average
    pairs = 0

    # calculate and sum all the uniquie pairwise distances
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):

            # calc the euclidean distance
            dist = calculate_distance(vectors[i], vectors[j])

            total_distance += dist
            pairs += 1

    # average distance across every pair we measured
    avg_pairwise_dist = total_distance / pairs

    return round(avg_pairwise_dist, 4)


# runs both genre_affinity_score and intra_list_diversity at once
def evaluate(recommendations: list, held_out_movies: list, affinity_function) -> dict:
    evaluations = {
        'genre_affinity_score': genre_affinity_score(recommendations, held_out_movies, affinity_function),
        'intra_list_diversity': intra_list_diversity(recommendations),
    }

    return evaluations


''' The following metrics below look at many users' recommendations together rather than
 just scoring one single recommendation list and all make use of the exposure dictionary'''

def build_exposure_dict(all_top_movies: list[list]) -> dict:
    #all_top_movies is a list of top_movies lists, one per user run

    exposure = {}

    # Going for each user's list
    for top_movies in all_top_movies:

        # Going for each movie in that user's list
        for movie, affinity in top_movies:
            # increment this movie's count by 1 or add to to the dictionary if it does not yet exist
            exposure[movie.id] = exposure.get(movie.id, 0) + 1

    # Returns {movie_id: number of times that movie was recommended to someone}
    return exposure

# how much of the entire catalog does the system recommend
def catalog_coverage(exposure: dict, n_catalog_movies: int) -> float:

    '''dividing the number of different movie ids in the exposure dictionary by the total number of movies in the
    whole catalog to get the the fraction of the catalog that ever got used'''

    coverage = len(exposure) / n_catalog_movies

    # rounding for easier reading
    return round(coverage , 4)

# Measures how evenly exposure is spread across the catalog
def gini_coefficient(exposure: dict, n_catalog_movies: int) -> float:

    # a list of zeros, accounting for movies that got exactly zero exposure(were never recommended to anyone)
    unexposed = [0] * (n_catalog_movies - len(exposure))

    counts = list(exposure.values()) + unexposed

    counts.sort()  # the Gini formula requires the counts to be sorted smallest to largest

    n = len(counts)
    total = sum(counts)

    # avoid dividing by zero if nothing was ever recommended to anyone
    if total == 0:
        return 0.0

    # this loop is part of the standard Gini coefficient formula.
    weighted_sum = 0
    for i, count in enumerate(counts):
        weighted_sum += (i + 1) * count

    # gini coefficient formula
    gini = (2 * weighted_sum) / (n * total) - (n + 1) / n

    return round(gini, 4)

# A second, different way of measuring how "spread out" the exposure is, borrowed from information theory.
def entropy_of_exposure(exposure: dict) -> float:
    total = sum(exposure.values())  # total number of recommendations handed out, across all movies

    # avoid dividing by zero if nothing was ever recommended
    if total == 0:
        return 0.0

    entropy = 0.0
    # go through each movie's exposure count
    for count in exposure.values():
        #this movie's portion of all recommendations handed out as a fraction

        portion = count / total

        # Using the base-2 logarithm, the standard choice for entropy calculations
        entropy -= portion * math.log2(portion)

    return round(entropy, 4)

# The following metrics compares recommendations across time or different rounds
def id_temporal_diversity(top_movies_a: list, top_movies_b: list) -> float:

    # taking the movie ids and ignoring their affinities to the antigen
    ids_a = set(movie.id for movie, affinity in top_movies_a)
    ids_b = set(movie.id for movie, affinity in top_movies_b)

    # combine every movie id seen in both rounds into a set
    union = ids_a | ids_b

    # if neither round recommended anything at all, there's nothing to compare
    if not union:
        return 0.0

    # count the ids that appear in both sets
    overlap = ids_a & ids_b

    ''' 1 minus how much overlap there was out of all the ids seen, so more overlap between the two
      means less diversity, and vice versa '''

    diversity = 1 - len(overlap) / len(union)

    return round(diversity, 4)


# measures temporal diversity on a genre vector level
def vector_temporal_diversity(top_movies_a: list, top_movies_b: list) -> float:

    vectors_a = set(tuple(movie.genres) for movie, affinity in top_movies_a)
    vectors_b = set(tuple(movie.genres) for movie, affinity in top_movies_b)

    # combine every genre vectors seen in both rounds into set
    union = vectors_a | vectors_b

    # if neither round recommended anything at all, there's nothing to compare
    if not union:
        return 0.0

    # get genre profiles that showed up in both round A and round B
    overlap = vectors_a & vectors_b

    # Jaccard distance used to measure how different two sets are
    jacc_distance = 1 - len(overlap) / len(union)

    return round(jacc_distance, 4)
