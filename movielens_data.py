''' script to get predictions for movielens data '''


from measures import predictions
from processing import preprocessing
import time
import pickle
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--movielens_data', choices=['small', '100k'], required=True)
    script_arguments = vars(parser.parse_args())
    movielens_data = script_arguments['movielens_data']

    if movielens_data == 'small':
        ratings = pickle.load(open("data/MOVIELENS/ml-latest-small/small_ratings_movielens.pkl","rb"))
        films = pickle.load(open("data/MOVIELENS/ml-latest-small/small_films_movielens.pkl","rb"))
    elif movielens_data == '100k':
        ratings = pickle.load(open("data/MOVIELENS/ml-100k/100k_benchmark_ratings.pkl","rb"))
        films = pickle.load(open("data/MOVIELENS/ml-100k/100k_benchmark_films_movielens.pkl","rb"))


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


    films, ratings_dict, compressed_test_ratings_dict, sims, movies_all_genres_matrix, movies_all_directors_matrix, movies_all_actors_matrix = preprocessing(ratings, films, movielens_data)
    start = time.time()

    MUR = 0.1
    MUG = 0.6
    MUA = 0.1
    MUD = 0.1

    nr_predictions, accuracy, rmse, mae, precision, recall, f1 = predictions(MUR, MUG, MUA, MUD, films, compressed_test_ratings_dict, ratings_dict, sims, movies_all_genres_matrix, movies_all_directors_matrix, movies_all_actors_matrix, movielens_data)

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


