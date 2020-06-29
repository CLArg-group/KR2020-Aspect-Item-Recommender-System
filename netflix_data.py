''' script to get predictions for netflix data '''


from measures import predictions
from processing import preprocessing
import time
import pickle



if __name__ == "__main__":
	ratings = pickle.load(open("data/NETFLIX/movie_ratings_500_id.pkl","rb"))
	films = pickle.load(open("data/NETFLIX/movie_metadata.pkl","rb"))

	films, ratings_dict, compressed_test_ratings_dict, sims, movies_all_genres_matrix, movies_all_directors_matrix, movies_all_actors_matrix = preprocessing(ratings, films, 'netflix')
	start = time.time()

	MUR = 0.1
	MUG = 0.8
	MUA = 0.1
	MUD = 0.1

	nr_predictions, accuracy, rmse, mae, precision, recall, f1 = predictions(MUR, MUG, MUA, MUD, films, compressed_test_ratings_dict, ratings_dict, sims, movies_all_genres_matrix, movies_all_directors_matrix, movies_all_actors_matrix, 'netflix')

	# print results
	print("Number of user-items pairs: %d" % nr_predictions)
	print("Accuracy: %.2f " % accuracy)
	print("RMSE: %.2f" % rmse)
	print("MAE: %.2f" % mae)
	print("Precision: %.2f" % precision)
	print("Recall: %.2f" % recall)
	print("F1: %.2f" % f1)

	end = time.time()
	print("\nComputing strengths took %d seconds" % (end-start))


