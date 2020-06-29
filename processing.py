''' functions to get movie aspects, compute similarity, get ratings to test '''


import time
import scipy
import numpy as np
import pandas as pd
from collections import Counter
import sklearn.preprocessing as pp
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor


THREADS = cpu_count() - 1


def map_aspect_values_to_movies(x):
	(film, meta), aspect = x
	aspects = dict()
	if aspect == "director" and type(meta[aspect]) is str:
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


def viewed_matrix(ratings_cold_start, all_films, data_origin):
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
		if data_origin == 'netflix':
			current_rating = int(train_ratings[i])
		elif data_origin == 'small':
			current_rating = float(train_ratings[i])
		elif data_origin == '100k':
			current_rating = int(train_ratings[i])

		try:
			movies_watched[current_item_id + "_" + all_films[current_item_id]["title"]][current_user_id] = current_rating
		except Exception:
			print('item id missing %s' % current_item_id) ## possibly the movies lacking info such as actors which are discarded

	return movies_watched


def get_movies_aspect_matrix(films, aspect_type):
	aspects_associated_to_movies = dict_movie_aspect(films, aspect_type)
	movies_all_aspects_matrix = pd.DataFrame.from_dict(aspects_associated_to_movies, dtype='int64', orient='index')
	movies_all_aspects_matrix = movies_all_aspects_matrix.replace(np.nan, 0)
	aspects_in_db = movies_all_aspects_matrix.keys()
	print('We have %d %s (an example is %s)' % (len(aspects_in_db), aspect_type, aspects_in_db[0]))
	return aspects_in_db, movies_all_aspects_matrix


def preprocessing(ratings, films, data_origin):
	start = time.time()

	all_init_users_ratings = [ratings_for_film["user_id"] for film_id in list(ratings.keys()) for ratings_for_film in ratings[film_id]]
	print("\nTotal number of users initially: %d" % len(set(all_init_users_ratings)))
	counter = Counter(all_init_users_ratings)
	print("Number of users who rated more than 100: %d" % len({x : counter[x] for x in counter if counter[x] > 100}))
	print("Number of users who rated more than 50: %d" % len({x : counter[x] for x in counter if counter[x] > 50}))
	print("Number of users who rated more than 30: %d" % len({x : counter[x] for x in counter if counter[x] > 30}))
	print("Number of users who rated between 10 and 30: %d" % len({x : counter[x] for x in counter if counter[x] <= 30 and counter[x] > 10}))

	genres_in_db, movies_all_genres_matrix = get_movies_aspect_matrix(films, "genre")
	directors_in_db, movies_all_directors_matrix = get_movies_aspect_matrix(films, "director")
	actors_in_db, movies_all_actors_matrix = get_movies_aspect_matrix(films, "actors")
	print('We have %d total aspects' % (len(genres_in_db)+len(directors_in_db)+len(actors_in_db)))

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

	# compute similarity
	movies_watched = viewed_matrix(train_ratings_dict, films, data_origin)
	movies_watched = pd.DataFrame.from_dict(movies_watched, dtype='int64', orient='index').T
	movies_watched = movies_watched.replace(np.nan, 0)

	user_ids_in_matrix = movies_watched.index.values
	
	# normalize vectors and then calculate cosine values by determining the matrix product
	movies_watched = movies_watched.T
	movies_watched = scipy.sparse.csc_matrix(movies_watched.values)
	normalized_matrix_by_column = pp.normalize(movies_watched.tocsc(), norm='l2', axis=0)
	cosine_sims = normalized_matrix_by_column.T * normalized_matrix_by_column
	assert cosine_sims.shape[0] == cosine_sims.shape[1] == len(user_ids_in_matrix)

	# convert similarities computed to dict
	sims = dict()
	for i in user_ids_in_matrix:
		sims[i] = []
	cosine_sims = cosine_sims.todok().items()

	for ((row,col), sim) in cosine_sims:
		if row != col:
			sims[user_ids_in_matrix[row]].append((user_ids_in_matrix[col], sim))

	end = time.time()
	print("\nComputing similarity took %d seconds" % (end-start))

	# convert ratings to a format as follows (film_id, user_id): rating
	ratings_dict = dict()
	user_ids = train_ratings_dict["userID"]
	item_ids = train_ratings_dict["itemID"]
	train_ratings = train_ratings_dict["rating"]
	assert len(user_ids) == len(item_ids) == len(train_ratings)

	for i in range(len(item_ids)):
		current_user_id = user_ids[i]
		current_item_id = item_ids[i]
		if data_origin == 'netflix':
			current_rating = int(train_ratings[i])
		elif data_origin == 'small':
			current_rating = float(train_ratings[i])
		elif data_origin == '100k':
			current_rating = int(train_ratings[i])

		tuple_key = (current_item_id, current_user_id)

		if data_origin == 'netflix':
			ratings_dict[tuple_key] = int(current_rating)
		elif data_origin == 'small':
			ratings_dict[tuple_key] = float(current_rating)
		elif data_origin == '100k':
			ratings_dict[tuple_key] = int(current_rating)


	return films, ratings_dict, compressed_test_ratings_dict, sims, movies_all_genres_matrix, movies_all_directors_matrix, movies_all_actors_matrix
