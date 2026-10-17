from flask import Flask, render_template, request, redirect, url_for
#from werkzeug.wrappers import Response

from src.data_loader import get_movies, get_ratings, create_users
from src.models import Movie, MovieRating, User
from src.movie_recommender import recommend_for_user
from src import aiNet
from src import knn_recommender
from src import evaluation

import random
import io
import base64
import functools
import threading

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

app = Flask(__name__)

GENRE_NAMES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

AFFINITY_FUNCTIONS = {
    'cosine': aiNet.cosine_sim_affinity,
    'euclidean': aiNet.euclid_distance_affinity,
    'dot': aiNet.dot_prod_affinity,
}

N_EXISTING_USERS = 943
N_MOVIES_TO_RATE = 10
KNN_K = 20

# smaller sample size cycle counts used specifically for the automatic background refresh triggered by rating movies or submitting parameters
LIVE_SAMPLE_SIZE = 25
LIVE_N_CYCLES = 5
LIVE_RATINGS_PER_CYCLE = 5

# load the movies and the full rating history
movies = get_movies('data/ml-100k/u.item')
movie_ratings = get_ratings('data/ml-100k/u.data')
users = create_users(movie_ratings, N_EXISTING_USERS, movies)

# default aiNet settings
settings = {
    'population_size': 100,
    'iterations': 50,
    'n_best': 20,
    'n_clones': 10,
    'mutation_lb': -0.1,
    'mutation_ub': 0.1,
    'suppression_threshold': 0.15,
    'stimulation_threshold': 0.0,
    'top_n_movies': 10,
    'affinity_function': 'cosine',
    'gamma': 0.5,
    'user_type': 'existing',
    'existing_user_id': 1,
}

# configuration for the catalog coverage, Gini, entropy of exposure, and temporal diversity across simulated cycles metrics
compare_state = {
    'sample_size': 100,
    'n_cycles': 5,
    'ratings_per_cycle': N_MOVIES_TO_RATE,
    'coverage_metrics': None,
    'temporal_results': None,
}

# state for whichever user (guest or existing) is currently being shown
guest_state = {
    # every MovieRating the current user has given, historical + this session
    'movie_ratings': [],

    # keeping track of already rated movies
    'ratings': {},

    # (movie, affinity) pairs from the last time recommendations were generated
    'recommendations': [],

    # the current user's preference profile, shown in the table on the recommendations page
    'profile': [0.0] * 19,

    # the movies currently offered up to be rated
    'movies_to_rate': [],

    # genre vectors of the top recommended movies from the last round,
    # used to seed the next round's population instead of starting from scratch
    'population': [],

    # population snapshots at key checkpoints from the most recent aiNet run
    'snapshots': [],
}


# resolves the currently selected affinity function. 
def get_affinity_function():
    if settings['affinity_function'] == 'rbf':
        # fill the the gamma parameter
        return functools.partial(aiNet.rbf_affinity, gamma=settings['gamma'])

    return AFFINITY_FUNCTIONS.get(settings['affinity_function'], AFFINITY_FUNCTIONS['cosine'])


# runs the aiNet algorithm once for a user, using whatever settings are currently set
def run_ainet(user: User, initial_population) -> tuple[list[tuple[Movie, float]], list[list[float]], list[dict]]:
    affinity_function = get_affinity_function()
    mutation_bounds = (settings['mutation_lb'], settings['mutation_ub'])

    recommendations, next_population, snapshots = recommend_for_user(user, movies,
        affinity_function=affinity_function,
        population_size=settings['population_size'],
        n_iterations=settings['iterations'],
        n_best=settings['n_best'],
        n_clones=settings['n_clones'],
        suppression_threshold=settings['suppression_threshold'],
        stimulation_threshold=settings['stimulation_threshold'],
        top_n_movies=settings['top_n_movies'],
        mutation_bounds=mutation_bounds,
        initial_population=initial_population,
    )

    return recommendations, next_population, snapshots


# id of the existing user currently being shown, or None for a new user
def get_current_user_id():
    if settings['user_type'] == 'existing':
        return settings['existing_user_id']

    return None

# runs user based kNN for a single user, and other users as neighbour candidates
def run_knn(rated_movie_ids_dict: dict, rated_movie_ids_set: set, exclude_user_id) -> list[tuple[Movie, float]]:
    neighbour_candidates = []

    # the target user's own record must not be one of their neighbours, otherwise it leaks their held out ratings
    for user in users:
        if user.id != exclude_user_id:
            neighbour_candidates.append(user)

    top_movies = knn_recommender.recommend_knn(rated_movie_ids_dict, rated_movie_ids_set, neighbour_candidates, movies,
                                        k=KNN_K, top_n=settings['top_n_movies'])

    return top_movies
    
def run_recommendation_for(user: User) -> list[tuple[Movie, float]]:
    recommendations, next_population, snapshots = run_ainet(user, guest_state['population'])

    # Continual learning, the learned population is carried over to the next round
    guest_state['population'] = next_population
    guest_state['snapshots'] = snapshots

    return recommendations


def pick_movies_to_rate(n: int) -> list[Movie]:
    # make a list of movie ids the user has already rated
    already_rated_ids = []
    for rating in guest_state['movie_ratings']:
        already_rated_ids.append(rating.movie_id)

    # keep only the movies that are not in that list
    unrated = []
    for movie in movies:
        if movie.id not in already_rated_ids:
            unrated.append(movie)

    # if there are fewer unrated movies than the required n ones, use the number of the unrated ones available
    n = min(n, len(unrated))

    random_unrated_movies = random.sample(unrated, n)

    return random_unrated_movies


# to enforce bounds
def clamp(value: int, min_val: int, max_val: int) -> int:
    if value < min_val:
        return min_val
    if value > max_val:
        return max_val

    return value


# samples existing users once, then run both aiNet and kNN for each of those same sampled users
def run_catalog_coverage_metrics(sample_size: int) -> dict:

    sample_size = clamp(sample_size, 1, len(users))

    # get a random sample of users
    sampled_users = random.sample(users, sample_size)

    ainet_top_movies = []
    knn_top_movies = []

    # get recommendations for each user using both systems
    for sampled_user in sampled_users:
        # get recommendations from the aiNet
        ainet_recs, next_pop, snapshots = run_ainet(sampled_user, None)
        ainet_top_movies.append(ainet_recs)

        # get recommendations from the KNN
        user_ratings_dict = knn_recommender.build_ratings_dict(sampled_user.movie_ratings)

        knn_recs = run_knn(user_ratings_dict, sampled_user.rated_movie_ids, sampled_user.id)
        knn_top_movies.append(knn_recs)

    ainet_exposure = evaluation.build_exposure_dict(ainet_top_movies)
    knn_exposure = evaluation.build_exposure_dict(knn_top_movies)

    evaluations = {
        'sample_size': sample_size,
        'ainet': {
            'coverage': evaluation.catalog_coverage(ainet_exposure, len(movies)),
            'gini': evaluation.gini_coefficient(ainet_exposure, len(movies)),
            'entropy': evaluation.entropy_of_exposure(ainet_exposure),
        },
        'knn': {
            'coverage': evaluation.catalog_coverage(knn_exposure, len(movies)),
            'gini': evaluation.gini_coefficient(knn_exposure, len(movies)),
            'entropy': evaluation.entropy_of_exposure(knn_exposure),
        },
    }

    return evaluations

# simulate rounds 
def run_temporal_diversity(n_cycles: int, ratings_per_cycle: int) -> list[dict]:
    local_ratings = list(guest_state['movie_ratings'])
    already_rated_ids = set(rating.movie_id for rating in local_ratings)

    ainet_cycle_top_movies = []
    knn_cycle_top_movies = []
    initial_population = None

    for cycle in range(n_cycles):
        if cycle > 0:
            unrated = [movie for movie in movies if movie.id not in already_rated_ids]
            n_to_add = min(ratings_per_cycle, len(unrated))

            if n_to_add > 0:
                new_movies = random.sample(unrated, n_to_add)

                for movie in new_movies:
                    score = random.randint(1, 5)
                    local_ratings.append(MovieRating(0, movie.id, score))
                    already_rated_ids.add(movie.id)

        cycle_user = User(0, local_ratings, movies)

        ainet_top_movies, next_population, _ = run_ainet(cycle_user, initial_population)
        ainet_cycle_top_movies.append(ainet_top_movies)
        initial_population = next_population

        cycle_ratings_dict = knn_recommender.build_ratings_dict(local_ratings)
        knn_top_movies = run_knn(cycle_ratings_dict, already_rated_ids, get_current_user_id())
        knn_cycle_top_movies.append(knn_top_movies)

    results = []
    for i in range(len(ainet_cycle_top_movies) - 1):
        results.append({
            'cycle_pair': f'{i} to {i + 1}',

            'ainet_id_diversity': evaluation.id_temporal_diversity(ainet_cycle_top_movies[i], ainet_cycle_top_movies[i + 1]),

            'ainet_vector_diversity': evaluation.vector_temporal_diversity(ainet_cycle_top_movies[i], ainet_cycle_top_movies[i + 1]),

            'knn_id_diversity': evaluation.id_temporal_diversity(knn_cycle_top_movies[i], knn_cycle_top_movies[i + 1]),

            'knn_vector_diversity': evaluation.vector_temporal_diversity(knn_cycle_top_movies[i], knn_cycle_top_movies[i + 1]),
        })

    return results


#background auto refresh for the expensive Group 2  Group 3 metrics
coverage_lock = threading.Lock()
coverage_retrigger_needed = threading.Event()

temporal_lock = threading.Lock()
temporal_retrigger_needed = threading.Event()


def coverage_worker() -> None:
    while True:
        coverage_retrigger_needed.clear()
        compare_state['coverage_metrics'] = run_catalog_coverage_metrics(LIVE_SAMPLE_SIZE)

        if not coverage_retrigger_needed.is_set():
            break

    coverage_lock.release()


# starts a background recomputation of catalog coverage / Gini / entropy of exposure,
def trigger_coverage_refresh() -> None:
    if coverage_lock.acquire(blocking=False):
        threading.Thread(target=coverage_worker, daemon=True).start()
    else:
        coverage_retrigger_needed.set()


def temporal_worker() -> None:
    while True:
        temporal_retrigger_needed.clear()
        compare_state['temporal_results'] = run_temporal_diversity(LIVE_N_CYCLES, LIVE_RATINGS_PER_CYCLE)

        if not temporal_retrigger_needed.is_set():
            break

    temporal_lock.release()


# kicks off a background recomputation of temporal diversity
def trigger_temporal_refresh() -> None:
    if temporal_lock.acquire(blocking=False):
        threading.Thread(target=temporal_worker, daemon=True).start()
    else:
        temporal_retrigger_needed.set()


# update the profile and recommendations, also get a fresh random set of unrated movies
def refresh_state() -> None:
    # build a dict of movie_id with rating score pairs
    ratings_dict = {}

    for rating in guest_state['movie_ratings']:
        ratings_dict[rating.movie_id] = rating.rating_score

    guest_state['ratings'] = ratings_dict

    combined_user = User(0, guest_state['movie_ratings'], movies)

    guest_state['profile'] = combined_user.preference_profile
    guest_state['recommendations'] = run_recommendation_for(combined_user)

    guest_state['movies_to_rate'] = pick_movies_to_rate(N_MOVIES_TO_RATE)

    trigger_temporal_refresh()


@app.route('/', methods=['GET', 'POST'])
def parameters():
    error_message = None

    if request.method == 'POST':
        # get the submitted paramter settings
        settings['population_size'] = int(request.form.get('population_size'))
        settings['iterations'] = int(request.form.get('iterations'))
        settings['n_best'] = int(request.form.get('n_best'))
        settings['n_clones'] = int(request.form.get('n_clones'))
        settings['mutation_lb'] = float(request.form.get('mutation_lb'))
        settings['mutation_ub'] = float(request.form.get('mutation_ub'))
        settings['suppression_threshold'] = float(request.form.get('suppression_threshold'))
        settings['stimulation_threshold'] = float(request.form.get('stimulation_threshold'))
        settings['top_n_movies'] = int(request.form.get('top_n_movies'))
        settings['affinity_function'] = request.form.get('affinity_function', 'cosine')
        settings['gamma'] = float(request.form.get('gamma'))
        settings['user_type'] = request.form.get('user_type', 'new')

        # reset carried over state whenever parameters or user change
        guest_state['population'] = []
        guest_state['snapshots'] = []

        # if the user is existing in the dataset
        if settings['user_type'] == 'existing':
            user_id = int(request.form.get('user_id'))

            if user_id is None or not (1 <= user_id <= len(users)):
                error_message = "Please enter a valid user id between 1 and " + str(len(users)) + "."
                return render_template('parameters.html', settings=settings, error=error_message)

            settings['existing_user_id'] = user_id

            existing_user = users[user_id - 1]

            # building on top of the user history as they rate more movies
            guest_state['movie_ratings'] = list(existing_user.movie_ratings)
        else:
            # for the new user with no history yet
            guest_state['movie_ratings'] = []

        refresh_state()

        trigger_coverage_refresh()

        # Head over the recommendations page
        return redirect(url_for('recommendations'))

    return render_template('parameters.html', settings=settings, error=error_message)


@app.route('/recommendations')
def recommendations() -> str:
    # pair up each genre name with its value in the profile
    profile_rows = []
    for i in range(len(GENRE_NAMES)):
        profile_rows.append((GENRE_NAMES[i], guest_state['profile'][i]))

    return render_template('recommendations.html', movie_objects=guest_state['movies_to_rate'],
        ratings=guest_state['ratings'],
        recommendations=guest_state['recommendations'],
        profile_rows=profile_rows,
        user_type=settings['user_type'],
        existing_user_id=settings['existing_user_id'],
    )


@app.route('/plots')
def plots() -> str:
    snapshots = guest_state['snapshots']
    antigen = guest_state['profile']

    if not snapshots:
        return render_template('plots.html', plot_images=[])

    # collect all vectors across all snapshots and the antigen so PCA is fitted on the full space, keeping all plots on the same axes
    all_vectors = [antigen]
    for snapshot in snapshots:
        if snapshot['population']:
            all_vectors.extend(snapshot['population'])

    pca = PCA(n_components=2)
    pca.fit(all_vectors)

    antigen_2d = pca.transform([antigen])[0]

    plot_images = []

    for snapshot in snapshots:
        if not snapshot['population']:
            continue

        pop_2d = pca.transform(snapshot['population'])

        fig, ax = plt.subplots(figsize=(6, 5))

        ax.scatter(
            pop_2d[:, 0], pop_2d[:, 1],
            alpha=0.7,
            s=40,
            label='Antibodies (movie vectors)',
        )

        # plot the antigen
        ax.scatter(antigen_2d[0], antigen_2d[1],
            c='red', marker='*', s=250,
            zorder=5, label='User profile (antigen)',
        )

        ax.set_title(snapshot['label'])

        # position the legend
        ax.legend(loc="center left", bbox_to_anchor=(0.7, 1.10))
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)

        plot_images.append({
            'label': snapshot['label'],
            'img': img_base64,
            'best_affinity': snapshot['best_affinity'],
            'mean_affinity': snapshot['mean_affinity'],
            'mean_net_stimulation': snapshot['mean_net_stimulation'],
        })

    return render_template('plots.html', plot_images=plot_images)


@app.route('/compare', methods=['GET', 'POST'])
def compare():
    if request.method == 'POST':
        compare_state['sample_size'] = int(request.form.get('sample_size', compare_state['sample_size']))
        compare_state['n_cycles'] = int(request.form.get('n_cycles', compare_state['n_cycles']))
        compare_state['ratings_per_cycle'] = int(request.form.get('ratings_per_cycle', compare_state['ratings_per_cycle']))

        compare_state['coverage_metrics'] = run_catalog_coverage_metrics(compare_state['sample_size'])
        compare_state['temporal_results'] = run_temporal_diversity(compare_state['n_cycles'], compare_state['ratings_per_cycle'])

        return redirect(url_for('compare'))

    all_ratings = guest_state['movie_ratings']

    if len(all_ratings) < 5:
        return render_template('compare.html',
            error="Not enough ratings to run a comparison. Please rate at least 5 movies first.",
            sample_size=compare_state['sample_size'],
            n_cycles=compare_state['n_cycles'],
            ratings_per_cycle=compare_state['ratings_per_cycle'],
            coverage_metrics=compare_state['coverage_metrics'],
            temporal_results=compare_state['temporal_results'],
        )

    # 80/20 train/test split on the current user's accumulated ratings
    split_idx = max(1, int(len(all_ratings) * 0.8))
    train_ratings = all_ratings[:split_idx]
    test_ratings = all_ratings[split_idx:]

    # build a lookup so we can find a movie by its id
    movie_lookup = {}
    for movie in movies:
        movie_lookup[movie.id] = movie

    # held out relevant movies: only those rated >= 4 count as ground truth positives
    held_out_movies = []
    for r in test_ratings:
        if r.rating_score >= 4 and r.movie_id in movie_lookup:
            held_out_movies.append(movie_lookup[r.movie_id])

    # training user preference profile built only from training ratings
    train_user = User(0, train_ratings, movies)

    affinity_function = get_affinity_function()

    # aiNet recommendations always fresh wuth no starting population for a fair comparison
    ainet_recs, next_pop, snapshots = run_ainet(train_user, None)

    # kNN recommendations using the same training ratings
    train_ratings_dict = knn_recommender.build_ratings_dict(train_ratings)
    train_rated_ids = set(train_ratings_dict.keys())

    knn_recs = run_knn(train_ratings_dict, train_rated_ids, get_current_user_id())

    # evaluate both systems on the same held out set with the same metrics.
    ainet_metrics = evaluation.evaluate(ainet_recs, held_out_movies, affinity_function)
    knn_metrics = evaluation.evaluate(knn_recs, held_out_movies, affinity_function)

    return render_template('compare.html',
        error=None,
        ainet_metrics=ainet_metrics,
        knn_metrics=knn_metrics,
        n_train=len(train_ratings),
        n_test=len(test_ratings),
        k=KNN_K,
        affinity_function_name=settings['affinity_function'],
        sample_size=compare_state['sample_size'],
        n_cycles=compare_state['n_cycles'],
        ratings_per_cycle=compare_state['ratings_per_cycle'],
        coverage_metrics=compare_state['coverage_metrics'],
        temporal_results=compare_state['temporal_results'],
    )


@app.route('/rate', methods=['POST'])
def rate():
    for movie in guest_state['movies_to_rate']:
        field_name = 'rating_' + str(movie.id)
        rating_value = request.form.get(field_name)

        if rating_value is not None:
            rating_value = int(rating_value)

            if rating_value > 0:
                guest_state['movie_ratings'].append(MovieRating(0, movie.id, rating_value))

    # every click is a new exposure, update the profile and pull a fresh unrated batch of
    # movies for the next round
    refresh_state()

    return redirect(url_for('recommendations'))


if __name__ == '__main__':
    app.run(debug=True, threaded=True)