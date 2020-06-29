''' computing ratings using different baseline algorithms '''

import random
import numpy as np
my_seed = 0
random.seed(my_seed)
np.random.seed(my_seed)
from surprise import Dataset
from surprise import Reader
from surprise import KNNBasic
from surprise import KNNWithZScore
from surprise import SVD
from surprise import NMF
from surprise import SlopeOne
from surprise import CoClustering
from measures import *
from processing import preprocessing
from math import sqrt
import pandas as pd
import argparse
import pickle


# baselines algorithms to use
algos = ["KNNBasic", "KNNWithZScore", "SVD", "NMF", "SlopeOne", "CoClustering"]



def testing(algo, ratings_dict, compressed_test_ratings_dict, data_origin):
	rat_dict_format = dict()
	rat_dict_format['userID'] = []
	rat_dict_format['itemID'] = []
	rat_dict_format['rating'] = []
	for (fid, uid), rat in ratings_dict.items():
		rat_dict_format['userID'].append(uid)
		rat_dict_format['itemID'].append(fid)
		rat_dict_format['rating'].append(rat)
	assert len(rat_dict_format['userID']) == len(rat_dict_format['itemID']) == len(rat_dict_format['rating'])

	train_df = pd.DataFrame(rat_dict_format)
	train_reader = Reader(rating_scale=(1, 5))
	train_data = Dataset.load_from_df(train_df[['userID', 'itemID', 'rating']], train_reader) # columns must correspond to userID, itemID and ratings (in that order)

	trainset = train_data.build_full_trainset()
	algo.train(trainset)

	predictions = []
	for user_id, true_ratings in compressed_test_ratings_dict.items():
		if true_ratings:
			for (film_id, str_rating) in true_ratings:
				# pred = algo.predict(user_id, film_id)
				pred = algo.predict(user_id, film_id).est
				if data_origin == 'netflix':
					predictions.append((int(str_rating), pred))
				elif data_origin == 'small':
					predictions.append((float(str_rating), pred))
				elif data_origin == '100k':
					predictions.append((int(str_rating), pred))

	if data_origin == 'netflix':
		true_ratings = [x for (x,y) in predictions]
		predicted_ratings = [round(y) for (x,y) in predictions]
		p, r, f = binary_predictions(true_ratings, predicted_ratings)
		return len(predictions), arg_accuracy_int(predictions), sqrt(mean_squared_error(true_ratings, predicted_ratings)), mean_absolute_error(true_ratings, predicted_ratings), p, r, f
	elif data_origin == 'small':
		true_ratings = [x for (x,y) in predictions]
		predicted_ratings = [round_of_rating(y) for (x,y) in predictions]
		p, r, f = binary_predictions(true_ratings, predicted_ratings)
		return len(predictions), arg_accuracy_float(predictions), sqrt(mean_squared_error(true_ratings, predicted_ratings)), mean_absolute_error(true_ratings, predicted_ratings), p ,r ,f
	elif data_origin == '100k':
		true_ratings = [x for (x,y) in predictions]
		predicted_ratings = [round(y) for (x,y) in predictions]
		p, r, f = binary_predictions(true_ratings, predicted_ratings)

		return len(predictions), arg_accuracy_int(predictions), sqrt(mean_squared_error(true_ratings, predicted_ratings)), mean_absolute_error(true_ratings, predicted_ratings), p ,r ,f


def run_baselines(ratings_dict, compressed_test_ratings_dict, data_origin):
	for alg in algos:
		if alg == "KNNBasic":
			algo = KNNBasic()
		elif alg == "KNNWithZScore":
			algo = KNNWithZScore()
		elif alg == "SVD":
			algo = SVD()
		elif alg == "NMF":
			algo = NMF()
		elif alg == "SlopeOne":
			algo = SlopeOne()
		elif alg == "CoClustering":
			algo = CoClustering()

		if data_origin == 'netflix':
			nr_predictions, accuracy, rmse, mae, precision, recall, f1 = testing(algo, ratings_dict, compressed_test_ratings_dict, 'netflix')
		elif data_origin == 'small':
			nr_predictions, accuracy, rmse, mae, precision, recall, f1 = testing(algo, ratings_dict, compressed_test_ratings_dict, 'small')
		elif data_origin == '100k':
			nr_predictions, accuracy, rmse, mae, precision, recall, f1 = testing(algo, ratings_dict, compressed_test_ratings_dict, '100k')

		# print results
		print ("\n\nAlg %s" % alg)
		print("Number of user-items pairs: %d" % nr_predictions)
		print("Accuracy: %.2f " % accuracy)
		print("RMSE: %.2f" % rmse)
		print("MAE: %.2f" % mae)
		print("Precision: %.2f" % precision)
		print("Recall: %.2f" % recall)
		print("F1: %.2f" % f1)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_type', choices=['netflix', 'small', '100k'], required=True)
	script_arguments = vars(parser.parse_args())
	data_type = script_arguments['data_type']

	if data_type == 'netflix':
		ratings = pickle.load(open("data/NETFLIX/movie_ratings_500_id.pkl","rb"))
		films = pickle.load(open("data/NETFLIX/movie_metadata.pkl","rb"))
	elif data_type == 'small':
		ratings = pickle.load(open("data/MOVIELENS/ml-latest-small/small_ratings_movielens.pkl","rb"))
		films = pickle.load(open("data/MOVIELENS/ml-latest-small/small_films_movielens.pkl","rb"))
	elif data_type == '100k':
		ratings = pickle.load(open("data/MOVIELENS/ml-100k/100k_benchmark_ratings.pkl","rb"))
		films = pickle.load(open("data/MOVIELENS/ml-100k/100k_benchmark_films_movielens.pkl","rb"))


	if data_type != 'netflix':
    	# remove from ratings the missing films (that were missing info and hence were discarded)
		ids_to_del_rf = set(ratings.keys()).difference(set(films.keys()))
		ids_to_del_fr = set(films.keys()).difference(set(ratings.keys()))
		ids_to_del = ids_to_del_rf.union(ids_to_del_fr)

		corrected_ratings = dict()
		for x in ratings.keys():
		    if x not in ids_to_del:
		        curr_rats = []
		        for curr_rat in ratings[x]:
		            temp_dict = dict()
		            temp_dict['user_rating'] = curr_rat['user_rating']
		            temp_dict['user_rating_date'] = curr_rat['user_rating_date']
		            temp_dict['user_id'] = 'x'+curr_rat['user_id']
		            curr_rats.append(temp_dict)
		        corrected_ratings[x] = curr_rats
		ratings = corrected_ratings

		corrected_films = dict()
		for x in films.keys():
		    if x not in ids_to_del:
		        corrected_films[x] = films[x]
		films = corrected_films
		assert len(ratings) == len(films)

	_, ratings_dict, compressed_test_ratings_dict, _, _, _, _ = preprocessing(ratings, films, data_type)
	run_baselines(ratings_dict, compressed_test_ratings_dict, data_type)
