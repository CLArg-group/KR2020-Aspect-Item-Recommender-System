import sys
import sklearn.preprocessing as pp
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from measures import *
from scipy import spatial
from math import sqrt
import jsonlines
import operator
import scipy
import pickle
import numpy as np
import pandas as pd
import time
import json

THREADS = 16


def tuple_dict_from_ratings(data):
	ratings_dict = dict()
	for film_id in list(data.keys()):
		for ratings in data[film_id]:
			tuple_key = (film_id, ratings["user_id"])
			ratings_dict[tuple_key] = int(ratings["user_rating"])
	return ratings_dict


def map_aspect_values_to_movies(x):
	(film, meta), aspect = x
	aspects = dict()
	if aspect == "director":
		aspects[meta[aspect]] = 1
	else:
		for g in meta[aspect]:
			aspects[g] = 1
	return film, meta, aspects


def dict_movie_aspect(paper_films, aspect):
	paper_films_aspect_prepended = map(lambda e: (e, aspect), list(paper_films.items()))
	aspect_dict = dict()
	with ProcessPoolExecutor(max_workers=THREADS) as executor:
		results = executor.map(map_aspect_values_to_movies, paper_films_aspect_prepended)
	for film, meta, aspects in results:
		aspect_dict[film + "_" + meta["title"]] = aspects
	return aspect_dict


def map_user_profile_normalized(x):
	df, user, movies_aspect_values = x
	user_movies = df.loc[:, user]
	profile = user_movies.dot(movies_aspect_values)
	for name in list(movies_aspect_values.columns):
		mav = movies_aspect_values.loc[:, name]
		assert len(mav) == len(user_movies)
		seen = 0
		for i in range(len(mav)):
			if mav[i] != 0 and user_movies[i] != 0:
				seen += 1
		if seen != 0:
			profile[name] /= seen
	return user, profile.to_dict()


def users_movie_aspect_preferences(movies_aspect_values, movies_watched, users):
	df = pd.DataFrame.from_dict(movies_watched, orient='index')
	df = df.replace(np.nan, 0)
	users_aspects_prefs = dict()

	with ProcessPoolExecutor(max_workers=THREADS) as executor:
		results = executor.map(map_user_profile_normalized, [(df, user, movies_aspect_values) for user in users])
	for user, user_profile in results:
		users_aspects_prefs[user] = user_profile
	return users_aspects_prefs


def viewed_matrix(ratings_cold_start, all_films):
	user_ids = ratings_cold_start["userID"]
	item_ids = ratings_cold_start["itemID"]
	train_ratings = ratings_cold_start["rating"]
	assert len(user_ids) == len(item_ids) == len(train_ratings)

	movies_watched = dict()
	for uid in all_films.keys():
		movies_watched[uid + "_" + all_films[uid]["title"]] = dict()

	for i in range(len(item_ids)):
		current_user_id = user_ids[i]
		current_item_id = item_ids[i]
		current_rating = int(train_ratings[i])
		
		try:
			movies_watched[current_item_id + "_" + all_films[current_item_id]["title"]][current_user_id] = current_rating
		except Exception:
			print ('item id missing %s' % current_item_id) ## possibly the movies lacking info such as actors which are discarded
	return movies_watched


def filter_unseen_movies(movies_genres, movies_watched):
	seen_movie_genres = dict()
	for k, v in movies_watched.items():
		if movies_watched[k]:
			seen_movie_genres[k] = movies_genres[k]
	return seen_movie_genres


def user_prefs(movies_watched, movies_aspects, users, aspect_type):
	movies_aspects = filter_unseen_movies(movies_aspects, movies_watched)
	movies_aspects = pd.DataFrame.from_dict(movies_aspects, dtype='int64', orient='index')
	movies_aspects = movies_aspects.replace(np.nan, 0)
	return users_movie_aspect_preferences(movies_aspects, movies_watched, users) 


def user_sim(users_genres_prefs):
	users_genres_prefs = pd.DataFrame.from_dict(users_genres_prefs, orient='index')
	user_ids_in_matrix = users_genres_prefs.index.values
	users_genres_prefs = users_genres_prefs.T
	users_genres_prefs = scipy.sparse.csc_matrix(users_genres_prefs.values)
	normalized_matrix_by_column = pp.normalize(users_genres_prefs.tocsc(), norm='l2', axis=0)
	cosine_sims = normalized_matrix_by_column.T * normalized_matrix_by_column

	sims = dict()
	for i in user_ids_in_matrix:
		sims[i] = []
	cosine_sims = cosine_sims.todok().items()

	for ((row,col), sim) in cosine_sims:
		if row != col:
			sims[user_ids_in_matrix[row]].append((user_ids_in_matrix[col], sim))
	return sims


def film_strength(user_id, film_id, films, ratings, all_actors, all_directors, all_genres, all_similarities, testing_users_cold_start_for_user, movies_genres, movies_directors, movies_actors):
	nSimUsers = 20 # number of similar users to use
	users_actors_prefs = testing_users_cold_start_for_user["actors"]
	users_directors_prefs = testing_users_cold_start_for_user["directors"]
	users_genres_prefs = testing_users_cold_start_for_user["genres"]
	similarities_for_new_user = testing_users_cold_start_for_user["sims"]
	simsSorted = sorted(similarities_for_new_user, key = operator.itemgetter(1), reverse = True)
	sims = simsSorted[:nSimUsers]
	film = films[film_id]

	# mu constants for this user 
	MUR = 0.7
	MUG = 0.8
	MUA = 0.1
	MUD = 0.1

	# take an average of each of the the genre's average ratings
	nGenres = 0
	dGenres = 0
	if type(film['genre']) is str:
		film['genre'] = [film['genre']]
	for genre in film['genre']:
		aspect_value =  movies_genres.loc[genre].to_dict()
		movie_ids_with_aspect_value = [k.split("_")[0] for k,v in aspect_value.items() if v == 1]
		
		# get the average rating for each film of that genre and take an average
		nGenre = 0
		dGenre = 0
		for genrefilm in movie_ids_with_aspect_value:
			avg_rat = average_rating(sims, genrefilm, ratings)
			if avg_rat:
				dGenre += avg_rat
				nGenre += 1

		if nGenre > 0:
			avGenre = dGenre / nGenre
			cmbGenre = ((((users_genres_prefs[genre]- 1) / 2)-1) + (MUR*avGenre)) / (1+MUR) 
		else:
			cmbGenre = (((users_genres_prefs[genre]- 1) / 2)-1)

		dGenres += cmbGenre
		nGenres += 1

	if nGenres > 0:
		avgGenreRating = dGenres / nGenres
	else:
		avgGenreRating = 0

	# take an average of each of the the actor's average ratings
	nActors = 0
	dActors = 0
	if type(film['actors']) is str:
		film['actors'] = [film['actors']]
	for actor in film['actors']:
		aspect_value =  movies_actors.loc[actor].to_dict()
		movie_ids_with_aspect_value = [k.split("_")[0] for k,v in aspect_value.items() if v == 1]
		
		# get the average rating for each film with that actor and take an average
		nActor = 0
		dActor = 0
		for actorfilm in movie_ids_with_aspect_value:
			avg_rat = average_rating(sims, actorfilm, ratings)
			if avg_rat:
				dActor += avg_rat
				nActor += 1

		if nActor > 0:
			avActor = dActor / nActor
			cmbActor = ((((users_actors_prefs[actor]- 1) / 2)-1) + (MUR*avActor)) / (1+MUR) 
		else:
			cmbActor = (((users_actors_prefs[actor]- 1) / 2)-1)

		dActors += cmbActor
		nActors += 1


	if nActors > 0:
		avgActorRating = dActors / nActors
	else:
		avgActorRating = 0

	# take an average of each of the the director's average ratings
	nDirectors = 0
	dDirectors = 0
	if type(film['director']) is str:
		film['director'] = [film['director']]
	for director in film['director']:
		aspect_value =  movies_directors.loc[director].to_dict()
		movie_ids_with_aspect_value = [k.split("_")[0] for k,v in aspect_value.items() if v == 1]
		
		# get the average rating for each film with that actor and take an average
		nDirector = 0
		dDirector = 0
		for directorfilm in movie_ids_with_aspect_value:
			avg_rat = average_rating(sims, directorfilm, ratings)
			if avg_rat:
				dDirector += avg_rat
				nDirector += 1

		if nDirector > 0:
			avDirector = dDirector / nDirector
			cmbDirector = ((((users_directors_prefs[director]- 1) / 2)-1) + (MUR*avDirector)) / (1+MUR)
		else:
			cmbDirector = (((users_directors_prefs[director]- 1) / 2)-1)

		dDirectors += cmbDirector
		nDirectors += 1

	if nDirectors > 0:
		avgDirectorRating = dDirectors / nDirectors
	else:
		avgDirectorRating = 0

	# calculates the item strength
	avg_rat = average_rating(sims, film_id, ratings)

	if avg_rat is None:
		item_strength = ((MUG * avgGenreRating) + (MUA * avgActorRating)+ (MUD * avgDirectorRating)) / (MUG + MUA + MUD)
	else:
		item_strength = ((MUR * avg_rat) + (MUG * avgGenreRating) + (MUA * avgActorRating)+ (MUD * avgDirectorRating)) / (MUR + MUG + MUA + MUD)
	return (((item_strength + 1)*2)+1)


def average_rating(sims, film_id, ratings):
	# counts and totals for each type of aspect
	nRatings = 0
	dRatings = 0

	# cycles through each of the similar users
	for sim in sims:
		user_id = sim[0]
		similarity = sim[1]

		# if a rating exists by this user on the film
		if (film_id, user_id) in ratings.keys():
			user_rating = ratings[(film_id, user_id)]
			scaled_rating = ((user_rating - 1) / 2)-1
			dRatings += scaled_rating * similarity
			nRatings += 1
	if nRatings == 0:
		avg_rat = None
	else:
		avg_rat = dRatings / nRatings
	return avg_rat



if __name__ == "__main__":
	start = time.time()
	ratings = pickle.load(open("data/NETFLIX/movie_ratings_500_id.pkl","rb"))
	films = pickle.load(open("data/NETFLIX/movie_metadata.pkl","rb"))

	# create dict indexed by user for the rated movies
	user_movie_ratings = dict()
	for mid, uratings in ratings.items():
		for urating in uratings:
			uid = urating['user_id']
			if uid not in user_movie_ratings:
				user_movie_ratings[uid] = []
			user_movie_ratings[uid].append((mid, urating['user_rating']))

	train_ratings_dict = dict()
	train_ratings_dict["userID"] = []
	train_ratings_dict["itemID"] = []
	train_ratings_dict["rating"] = []
	compressed_test_ratings_dict = dict()

	# if user rated >30, use 30 movies for testing and the remaining for training
	# if user rated 10<=30, use 10 for testing and the remaining for training
	for umv, fratings in user_movie_ratings.items():
		if len(fratings) > 30:
			for i in range(len(fratings)-30):
				train_ratings_dict["userID"].append(umv)
			train_ratings_dict["itemID"].extend([m for (m,r) in fratings[30:]])
			train_ratings_dict["rating"].extend([r for (m,r) in fratings[30:]])
			compressed_test_ratings_dict[umv] = fratings[:30]
		elif len(fratings) <= 30 and len(fratings) > 10:
			for i in range(len(fratings)-10):
				train_ratings_dict["userID"].append(umv)
			train_ratings_dict["itemID"].extend([m for (m,r) in fratings[10:]])
			train_ratings_dict["rating"].extend([r for (m,r) in fratings[10:]])
			compressed_test_ratings_dict[umv] = fratings[:10]

	sample_users = set(train_ratings_dict["userID"])
	print ('NR USERS %d' % len(sample_users))

	movies_genres = dict_movie_aspect(films, "genre")
	movies_genres = pd.DataFrame.from_dict(movies_genres, dtype='int64', orient='index').T
	movies_genres = movies_genres.replace(np.nan, 0)

	movies_directors = dict_movie_aspect(films, "director")
	movies_directors = pd.DataFrame.from_dict(movies_directors, dtype='int64', orient='index').T
	movies_directors = movies_directors.replace(np.nan, 0)

	movies_actors = dict_movie_aspect(films, "actors")
	movies_actors = pd.DataFrame.from_dict(movies_actors, dtype='int64', orient='index').T
	movies_actors = movies_actors.replace(np.nan, 0)

	movies_watched = viewed_matrix(train_ratings_dict, films)

	# compute preferences
	print (f'Begin prefs')
	all_actors = user_prefs(movies_watched, movies_actors, sample_users, "actors")
	print (f'End actors')
	all_directors = user_prefs(movies_watched, movies_directors, sample_users, "director")
	print (f'End directors')
	all_genres = user_prefs(movies_watched, movies_genres, sample_users, "genre")
	print (f'End genres')
	all_similarities = user_sim(all_genres)

	ratings = tuple_dict_from_ratings(ratings)

	testing_users_cold_start = dict()
	for user in compressed_test_ratings_dict.keys():
		testing_users_cold_start[user] = dict()
		testing_users_cold_start[user]["actors"] = all_actors[user]
		testing_users_cold_start[user]["directors"] = all_directors[user]
		testing_users_cold_start[user]["genres"] = all_genres[user]
		testing_users_cold_start[user]["sims"] = all_similarities[user]

	predictions = []
	for user_id, true_ratings in compressed_test_ratings_dict.items():
		if true_ratings:
			for (film_id, str_rating) in true_ratings:
				strength = film_strength(user_id, film_id, films, ratings, all_actors, all_directors, all_genres, all_similarities, testing_users_cold_start[user_id], movies_genres, movies_directors, movies_actors)
				predictions.append((int(str_rating), strength))

	true_ratings = [x for (x,y) in predictions]
	predicted_ratings = [round(y) for (x,y) in predictions]
	p, r, f = binary_predictions(true_ratings, predicted_ratings)

	print(f'Number of user-items pairs: {len(predictions)}')
	print(f'Accuracy: {arg_accuracy_int(predictions)}')
	print(f'RMSE {sqrt(mean_squared_error(true_ratings, predicted_ratings))}')
	print(f'MAE {mean_absolute_error(true_ratings, predicted_ratings)}')
	print(f'Precision: {p}')
	print(f'Recall: {r}')
	print(f'F1: {f}')

	end = time.time()
	print(f'time {end-start}')



	