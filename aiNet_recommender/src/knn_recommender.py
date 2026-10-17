import math
from .models import Movie, User

# build {movie_id: rating_score} dict from a list of MovieRating objects
def build_ratings_dict(movie_ratings: list) -> dict:
    ratings_dict = {}

    for rating in movie_ratings:
        ratings_dict[rating.movie_id] = rating.rating_score

    return ratings_dict


# cosine similarity between two users based on movies they both rated
def cosine_similarity(ratings_a: dict, ratings_b: dict, min_common: int = 2) -> float:
    # get movies both users have rated
    common = set(ratings_a.keys()) & set(ratings_b.keys())

    # users must've rated atleast a few common movies
    if len(common) < min_common:
        return 0.0

    # compute the dot product of the rating scores
    dot_prod = 0.0
    for movie_id in common:
        dot_prod += ratings_a[movie_id] * ratings_b[movie_id]

    # compute the norms of the two ratings vectors
    norm_a = 0.0
    norm_b = 0.0

    for movie_id  in common:
        norm_a += ratings_a[movie_id] ** 2
        norm_b += ratings_b[movie_id] ** 2

    norm_a, norm_b = math.sqrt(norm_a), math.sqrt(norm_b)

    # to avoid division by zero
    if norm_a == 0 or norm_b == 0:
        return 0.0

    cosine_sim = dot_prod/ (norm_a * norm_b)

    return cosine_sim

# find the k most similar users to the target user by cosine similarity
def find_neighbours(target_ratings: dict, all_users: list, k: int) -> list:
    similarities = []

    for user in all_users:
        user_ratings = build_ratings_dict(user.movie_ratings)
        similarity = cosine_similarity(target_ratings, user_ratings)

        # capture users with positive similarity only
        if similarity > 0:
            similarities.append((user, similarity))

    # sort in descending order by similarity score
    similarities.sort(key=lambda k: k[1], reverse=True)

    top_k = similarities[:k]

    return top_k


# predict a rating for a movie using a method known as similarity weighted average
def predict_rating(movie_id: str, neighbours: list) -> float:
    numerator = 0.0
    denominator = 0.0

    for neighbour, similarity in neighbours:
        neighbour_ratings = build_ratings_dict(neighbour.movie_ratings)

        # only if the neighbours have rated the movie
        if movie_id in neighbour_ratings:
            numerator += similarity * neighbour_ratings[movie_id]
            denominator += similarity

    if denominator == 0:
        return 0.0

    return numerator / denominator

# run user based knn recommendation
def recommend_knn(target_ratings: dict, rated_movie_ids: set, all_users: list[User], movies: list[Movie], k: int, top_n: int) -> list:
    neighbours = find_neighbours(target_ratings, all_users, k)

    if not neighbours:
        return []

    # only consider movies the target user has not already rated
    candidate_movies = []
    for movie in movies:
        if movie.id not in rated_movie_ids:
            candidate_movies.append(movie)

    scored = []
    for movie in candidate_movies:
        predicted = predict_rating(movie.id, neighbours)

        # if the user would like the movie
        if predicted > 0:
            scored.append((movie, predicted))

    scored.sort(key=lambda x: x[1], reverse=True)

    top_scored = scored[:top_n]

    return top_scored