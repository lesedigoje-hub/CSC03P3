import math

# build {movie_id: rating_score} dict from a list of MovieRating objects
def build_ratings_dict(movie_ratings) -> dict:
    return {rating.movie_id: rating.rating_score for rating in movie_ratings}


# cosine similarity between two users based on movies they both rated
def cosine_similarity(ratings_a: dict, ratings_b: dict, min_common: int = 2) -> float:
    common = set(ratings_a.keys()) & set(ratings_b.keys())

    if len(common) < min_common:
        return 0.0

    dot = sum(ratings_a[m] * ratings_b[m] for m in common)
    norm_a = math.sqrt(sum(ratings_a[m] ** 2 for m in common))
    norm_b = math.sqrt(sum(ratings_b[m] ** 2 for m in common))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)

# find the k most similar users to the target user by cosine similarity
def find_neighbours(target_ratings: dict, all_users, k: int) -> list:
    similarities = []

    for user in all_users:
        user_ratings = build_ratings_dict(user.movie_ratings)
        sim = cosine_similarity(target_ratings, user_ratings)

        if sim > 0:
            similarities.append((user, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:k]


# predict a rating for a movie as a similarity weighted average of neighbour ratings
def predict_rating(movie_id: str, neighbours: list) -> float:
    numerator = 0.0
    denominator = 0.0

    for neighbour, similarity in neighbours:
        neighbour_ratings = build_ratings_dict(neighbour.movie_ratings)

        if movie_id in neighbour_ratings:
            numerator += similarity * neighbour_ratings[movie_id]
            denominator += similarity

    if denominator == 0:
        return 0.0

    return numerator / denominator

# run user based knn recommendation
def recommend_knn(target_ratings: dict, rated_movie_ids: set, all_users, movies, k: int, top_n: int) -> list:
    neighbours = find_neighbours(target_ratings, all_users, k)

    if not neighbours:
        return []

    # only consider movies the target user has not already rated
    candidate_movies = [movie for movie in movies if movie.id not in rated_movie_ids]

    scored = []
    for movie in candidate_movies:
        predicted = predict_rating(movie.id, neighbours)

        if predicted > 0:
            scored.append((movie, predicted))

    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_n]