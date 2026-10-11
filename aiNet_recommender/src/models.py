class Movie:
    def __init__(self, name, movie_id, genres):
        self.name = name
        self.id = movie_id

        ''' in the order (unknown, Action, Adventure, Animation, Children's, Comedy, Crime, Documentary, Drama, Fantasy,
                          Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western) '''
        self.genres = genres


class MovieRating:
    def __init__(self, user_id, movie_id, rating_score):
        self.user_id = user_id
        self.movie_id = movie_id
        self.rating_score = rating_score


class User:
    def __init__(self, user_id, movie_ratings, movie_list):
        self.id = user_id
        self.movie_ratings = movie_ratings
        self.preference_profile = self.create_preference_profile(movie_list)
        self.rated_movie_ids = set()

        # keep track of which movies this user has already rated, so they don't get recommended the same movies they've seen
        for rating in self.movie_ratings:
            self.rated_movie_ids.add(rating.movie_id)

    # Helper methods

    def get_movie_from_rating(self, movie_rating: MovieRating, movie_lookup: dict) -> Movie:
        return movie_lookup.get(movie_rating.movie_id)

    def create_preference_profile(self, movie_list: list[Movie]):

        movie_lookup = {movie.id: movie for movie in movie_list}

        # start with a total rating score of zero for each profile
        total_rating_scores = [0.0 for i in range(19)]

        # capture the numbers of movies that fall under a genre( to be used for calculating average scores)
        movies_per_genre = [0 for i in range(19)]

        for rating in self.movie_ratings:
            movie = self.get_movie_from_rating(rating, movie_lookup)

            if movie is None:
                continue

            for i in range(len(movie.genres)):
                if movie.genres[i] == 1:

                    # add rating score to the total for the genre
                    total_rating_scores[i] += rating.rating_score
                    movies_per_genre[i] += 1

        # get average rating scores
        average_rating_scores = [0.0 for i in range(19)]

        for i in range(19):
            if movies_per_genre[i] > 0:
                average_rating_scores[i] = total_rating_scores[i] / movies_per_genre[i]
            else:
                average_rating_scores[i] = 0.0

        # normalize the rating scores to 0 <= score <= 1 rounded to three decimal places
        preference_profile = [round(x / 5, 3) for x in average_rating_scores]

        return preference_profile
