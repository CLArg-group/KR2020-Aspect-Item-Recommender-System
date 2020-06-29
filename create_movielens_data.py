''' create datasets given the required format used in experiments '''


import csv
import json
from bs4 import BeautifulSoup
import urllib.request


DATA_DIRECTORY = '../data/MOVIELENS/ml-20m'
RATINGS_FILE = DATA_DIRECTORY + '/ratings.csv'  ## userId,movieId,rating,timestamp
MOVIES_FILE = DATA_DIRECTORY + '/movies.csv'    ## movieId,title,genres
LINKS_FILE = DATA_DIRECTORY + '/links.csv'     ## movieId,imdbId,tmdbId


DATA_DIRECTORY_BENCHMARK = '../data/MOVIELENS/ml-100k'
GENRE_FILE = DATA_DIRECTORY_BENCHMARK + '/u.genre'  ## Genre|genre_index
RATINGS_FILE = DATA_DIRECTORY_BENCHMARK + '/u.data'  ## tab separated user id | item id | rating | timestamp
MOVIES_FILE = DATA_DIRECTORY_BENCHMARK + '/u.item'    ## movie id | movie title | release date | video release date | IMDb URL | g1 | g2 |
LINKS_FILE = DATA_DIRECTORY_BENCHMARK + '/movie_url.csv'     ## movieId,imdb_url


class MovielensBenchmark():
    def __init__(self):
        self.genre_dict = self.get_benchmark_genre()
        self.links_dict = self.get_benchmark_links()


    def get_benchmark_genre(self):
        genre_dict = dict()
        with open(GENRE_FILE) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='|')
            for row in csv_reader:
                genre_dict[int(row[1])] = row[0]

        return genre_dict


    def get_benchmark_links(self):
        links_dict = dict()
        with open(LINKS_FILE) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                links_dict[row[0]] = row[1].split('/')[4]

        return links_dict


    def load_ratings(self):
        ratings = dict()

        with open(RATINGS_FILE) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            next(csv_reader, None) ## skip header

            for row in csv_reader:
                userId = row[0] ; movieId = row[1] ; rating = row[2] ; timestamp = row[3]
                if movieId not in ratings.keys():
                    ratings[movieId] = []
                current_rating = dict()
                current_rating['user_rating'] = rating
                current_rating['user_rating_date'] = timestamp
                current_rating['user_id'] = userId
                ratings[movieId].append(current_rating)

        return ratings


    def load_films(self):
        films = dict()

        with open(MOVIES_FILE) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='|')

            for row in csv_reader:
                movieId = row[0] ; title = row[1] ; genres_hot = row[-19:]

                genres = []
                for i in range(len(genres_hot)):
                    if genres_hot[i] == "1":
                        genres.append(self.genre_dict[i])

                try:
                    imdb_id = self.links_dict[movieId]

                    api_imdb = f"https://api.themoviedb.org/3/find/%s?api_key={api_key}&language=en-US&external_source={imdb_id}"
                    with urllib.request.urlopen(api_imdb) as url:
                        data = json.loads(url.read().decode())
                    tmdbId = str(data['movie_results'][0]['id'])

                    directors_names, actors_names = self.get_directors_actors('https://www.themoviedb.org/movie/' + tmdbId)

                    if directors_names and actors_names:
                        films[movieId] = dict()
                        films[movieId]['director'] = directors_names
                        films[movieId]['actors'] = actors_names
                        films[movieId]['title'] = title
                        films[movieId]['genre'] = genres
                except Exception:
                    print('Missing link for movie id %s' % movieId)

        return films

    def get_directors_actors(self, weblink):
        content = urllib.request.urlopen(weblink).read()
        soup = BeautifulSoup(content, 'lxml')
        directors_names = []
        actors_names = []

        directors = soup.find_all('ol', class_='people no_image')[0].find_all('li')
        for director in directors:
            director_html = director.find_all('p')
            assert len(director_html) == 2
            if  'Director' in director_html[1].text.strip():
                directors_names.append(director_html[0].text.strip())

        try:
            actors_names = soup.find_all("ol", class_="people scroller")[0].find_all("a")
            actors_names = [x.text.strip() for x in actors_names if x] ## includes images
            actors_names = [x for x in actors_names if x] ## filtered out images
        except Exception:
            return None, None ## no info about actors for some movies

        return directors_names, actors_names




class Movielens():

    def __init__(self):
        self.links_dict = self.get_links()


    def get_links(self):
        links_dict = dict()
        with open(LINKS_FILE) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader, None) ## skip header

            for row in csv_reader:
                links_dict[row[0]] = row[2]

        return links_dict


    def load_ratings(self):
        ratings = dict()

        with open(RATINGS_FILE) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader, None) ## skip header

            for row in csv_reader:
                userId = row[0] ; movieId = row[1] ; rating = row[2] ; timestamp = row[3]
                if movieId not in ratings.keys():
                    ratings[movieId] = []
                current_rating = dict()
                current_rating['user_rating'] = rating
                current_rating['user_rating_date'] = timestamp
                current_rating['user_id'] = userId
                ratings[movieId].append(current_rating)

        return ratings


    def load_films(self):
        films = dict()

        with open(MOVIES_FILE) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader, None) ## skip header

            for row in csv_reader:
                movieId = row[0] ; title = row[1] ; genres = row[2].split('|')

                try:
                    directors_names, actors_names = self.get_directors_actors('https://www.themoviedb.org/movie/' + self.links_dict[movieId])

                    if directors_names and actors_names:
                        films[movieId] = dict()
                        films[movieId]['director'] = directors_names
                        films[movieId]['actors'] = actors_names
                        films[movieId]['title'] = title
                        films[movieId]['genre'] = genres
                except Exception:
                    print('Missing link for movie id %s' % movieId)

        return films

    def get_directors_actors(self, weblink):
        content = urllib.request.urlopen(weblink).read()
        soup = BeautifulSoup(content, 'lxml')
        directors_names = []
        actors_names = []

        directors = soup.find_all('ol', class_='people no_image')[0].find_all('li')
        for director in directors:
            director_html = director.find_all('p')
            assert len(director_html) == 2
            if  'Director' in director_html[1].text.strip():
                directors_names.append(director_html[0].text.strip())

        try:
            actors_names = soup.find_all("ol", class_="people scroller")[0].find_all("a")
            actors_names = [x.text.strip() for x in actors_names if x] ## includes images
            actors_names = [x for x in actors_names if x] ## filtered out images
        except Exception:
            return None, None ## no info about actors for some movies

        return directors_names, actors_names


def create_data_in_format():
    movielens = Movielens()

    ratings = movielens.load_ratings()
    afile = open("ratings.pkl", "wb")
    pickle.dump(ratings, afile)
    afile.close()

    films = movielens.load_films()
    afile = open("films_movielens.pkl", "wb")
    pickle.dump(films, afile)
    afile.close()

    
    movielens = MovielensBenchmark()

    ratings = movielens.load_ratings()
    afile = open("ratings.pkl", "wb")
    pickle.dump(ratings, afile)
    afile.close()

    films = movielens.load_films()
    afile = open("films_movielens.pkl", "wb")
    pickle.dump(films, afile)
    afile.close()