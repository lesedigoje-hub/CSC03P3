from .models import Movie, MovieRating, User


def get_movies(filename: str) -> list[Movie]:
    file = open(filename, 'r', encoding='ISO-8859-1')

    # get list of lines(each representing info about one movie)
    movie_lines = file.readlines()

    movies = []
    for movie in movie_lines:
        line_split = movie.split('|')

        # get movie id
        movie_id = line_split[0]

        # get movie name
        movie_name = line_split[1]

        # get genre vector
        list_genres = line_split[-19:]
        list_genres[18] = list_genres[18][0]  # remove the EOL('\n') on the last element

        # convert genres to integers
        list_genres = [int(x) for x in list_genres]

        # create movie instance
        movie = Movie(movie_name, movie_id, list_genres)

        # add to list of movies
        movies.append(movie)

    file.close()
    return movies


def get_ratings(filename: str) -> list[MovieRating]:
    file = open(filename, 'r', encoding='ISO-8859-1')

    # get ratings line by line
    rating_lines = file.readlines()

    movie_ratings = []

    for line in rating_lines:
        line_split = line.split('\t')

        # get user id
        user_id = int(line_split[0])

        # get movie id
        movie_id = line_split[1]

        # get rating score
        rating_score = int(line_split[2])

        # create movie rating instance
        movie_rating = MovieRating(user_id, movie_id, rating_score)

        movie_ratings.append(movie_rating)

    file.close()
    return movie_ratings


def create_users(movie_ratings: list[MovieRating], n_users: int, movie_list) -> list[User]:
    # group ratings by user id once, instead of re scanning all ratings per user
    ratings_by_user = {}
    for rating in movie_ratings:
        ratings_by_user.setdefault(rating.user_id, []).append(rating)

    users = []

    for i in range(n_users):
        # create a new user id
        user_id = i + 1

        ratings = ratings_by_user.get(user_id, [])

        # create user instance
        user = User(user_id, ratings, movie_list)
        users.append(user)

    return users
