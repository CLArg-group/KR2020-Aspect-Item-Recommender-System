# Recommender Systems for Interactive Explanations

Source code, data, and supplementary materials for Recommender Systems for Interactive Explanations.

## Project structure

`data` - data for experiments

`movielens_data.py` - main file to run experiments for movielens data

`netflix_data.py` - main file to run experiments for netflix data

`aspect_item_rs.py` - contains the algorithm for the A-I* RS.

`baseline.py` - contains the code for the baselines used in experiments

`processing.py` - contains functions to get movie aspects, to compute similarity, to get ratings used for testing

`compute_strength.py` - contains the function that calculates the strength of the film

`measures.py` - contains the functions that determine the performance of the system

`create_movielens_data.py` creates and filters movielens data in the required format (an API key from TMDB is required to get information about actors and directors)

## Data

Each source has 2 files, one for ratings and one for films.

Each entry in the `ratings` dictionary is in the following format:

```
[
  {'user_rating': '...', 'user_rating_date': '...', 'user_id': '...'}, 
  {'user_rating': '...', 'user_rating_date': '...', 'user_id': '...'},
  ...
]
```

Each entry in the `films` dictionary is in the following format:

```
{
  'director': [...], 
  'actors': [...], 
  'title': '...', 
  'genre': [...]
}
```

## Experiments

How to run the scripts for the 2 sources (movielens and netflix):

```
python3 movielens_data.py --movielens_data 100k
python3 movielens_data.py --movielens_data small
python3 netflix_data.py
```

How to obtain baselines results:

`python3 baseline.py --data_type data` (where data can be one of the following values: `netflix`, `small`, `100k`)

For example, running `python3 movielens_data.py --movielens_data 100k` should give the following output:

```
Total number of users initially: 943
Number of users who rated more than 100: 312
Number of users who rated more than 50: 510
Number of users who rated more than 30: 689
Number of users who rated between 10 and 30: 254
We have 19 genre (an example is Animation)
We have 867 director (an example is John Lasseter)
We have 3442 actors (an example is Tom Hanks)
We have 4328 total aspects

Computing similarity took 2 seconds
Number of user-items pairs: 23210
Accuracy: 0.84
RMSE: 1.10
MAE: 0.84
Precision: 0.87
Recall: 0.98
F1: 0.92

Computing strengths took 93 seconds
```
