''' calculate film strength'''


import operator


def film_strength(MUR, MUG, MUA, MUD, user_id, film_id, films, ratings, similarities_for_user, movies_genres, movies_directors, movies_actors):
	nSimUsers = 20 # number of similar users to use
	simsSorted = sorted(similarities_for_user, key = operator.itemgetter(1), reverse = True)
	sims = simsSorted[:nSimUsers]
	film = films[film_id]

	# take an average of each of the the genre's average ratings
	nGenres = 0
	dGenres = 0
	if type(film['genre']) is str:
		film['genre'] = [film['genre']]
	for genre in film['genre']:
		aspect_value =  movies_genres[genre].to_dict()
		movie_ids_with_aspect_value = [k.split("_")[0] for k,v in aspect_value.items() if v == 1]
		
		# get the average rating for each film of this genre and take an average of those from the user and similar users
		nGenre = 0
		dGenre = 0
		nGenreSim = 0
		dGenreSim = 0

		for genrefilm in movie_ids_with_aspect_value:
			if (genrefilm, user_id) in ratings.keys():
				dGenre += ((ratings[(genrefilm, user_id)] - 1) / 2)-1   # adds this to the current user's ratings total for this genre
				nGenre += 1         # and the count
			else:
				avg_rat = average_rating(sims, genrefilm, ratings)
				if avg_rat:
					dGenreSim += MUR * avg_rat  # adds this average to the similar users' ratings total for this genre
					nGenreSim += 1       # and the count

		if nGenre > 0:              # if we have films of this genre with ratings from the user
			if nGenreSim > 0:        # and also films of this genre with ratings from similar users
				avGenre = ((dGenre / nGenre) + (dGenreSim / nGenreSim)) / (1 + MUR)       # uses both the current user's and similar users' ratings 
			else:
				avGenre = dGenre / nGenre       # uses only the current user's ratings
		else:                       # if we do not have films of this genre with ratings from the user
			if nGenreSim > 0:        # but we have films of this genre with ratings from similar users
				avGenre = dGenreSim / nGenreSim       # uses only the similar users' ratings
			else:
				avGenre = 0

		dGenres += avGenre
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
		aspect_value =  movies_actors[actor].to_dict()
		movie_ids_with_aspect_value = [k.split("_")[0] for k,v in aspect_value.items() if v == 1]

		# get the average rating for each film of this actor and take an average of those from the user and similar users
		nActor = 0
		dActor = 0
		nActorSim = 0
		dActorSim = 0

		for actorfilm in movie_ids_with_aspect_value:
			if (actorfilm, user_id) in ratings.keys():
				dActor += ((ratings[(actorfilm, user_id)] - 1) / 2)-1   # adds this to the current user's ratings total for this actor
				nActor += 1         # and the count
			else:
				avg_rat = average_rating(sims, actorfilm, ratings)
				if avg_rat:
					dActorSim += MUR * avg_rat  # adds this average to the similar users' ratings total for this actor
					nActorSim += 1       # and the count

		if nActor > 0:              # if we have films of this actor with ratings from the user
			if nActorSim > 0:        # and also films of this actor with ratings from similar users
				avActor = ((dActor / nActor) + (dActorSim / nActorSim)) / (1 + MUR)       # uses both the current user's and similar users' ratings 
			else:
				avActor = dActor / nActor       # uses only the current user's ratings
		else:                       # if we do not have films of this actor with ratings from the user
			if nActorSim > 0:        # but we have films of this actor with ratings from similar users
				avActor = dActorSim / nActorSim       # uses only the similar users' ratings
			else:
				avActor = 0

		dActors += avActor
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
		aspect_value =  movies_directors[director].to_dict()
		movie_ids_with_aspect_value = [k.split("_")[0] for k,v in aspect_value.items() if v == 1]

		# get the average rating for each film of this director and take an average of those from the user and similar users
		nDirector = 0
		dDirector = 0
		nDirectorSim = 0
		dDirectorSim = 0

		for directorfilm in movie_ids_with_aspect_value:
			if (directorfilm, user_id) in ratings.keys():
				dDirector += ((ratings[(directorfilm, user_id)] - 1) / 2)-1   # adds this to the current user's ratings total for this director
				nDirector += 1         # and the count
			else:
				avg_rat = average_rating(sims, directorfilm, ratings)
				if avg_rat:
					dDirectorSim += MUR * avg_rat  # adds this average to the similar users' ratings total for this Director
					nDirectorSim += 1       # and the count

		if nDirector > 0:              # if we have films of this Director with ratings from the user
			if nDirectorSim > 0:        # and also films of this Director with ratings from similar users
				avDirector = ((dDirector / nDirector) + (dDirectorSim / nDirectorSim)) / (1 + MUR)       # uses both the current user's and similar users' ratings 
			else:
				avDirector = dDirector / nDirector       # uses only the current user's ratings
		else:                       # if we do not have films of this Director with ratings from the user
			if nDirectorSim > 0:        # but we have films of this Director with ratings from similar users
				avDirector = dDirectorSim / nDirectorSim       # uses only the similar users' ratings
			else:
				avDirector = 0

		dDirectors += avDirector
		nDirectors += 1

	if nDirectors > 0:
		avgDirectorRating = dDirectors / nDirectors
	else:
		avgDirectorRating = 0

	# compute strength
	item_strength = ((MUG * avgGenreRating) + (MUA * avgActorRating)+ (MUD * avgDirectorRating)) / (MUG + MUA + MUD)
	film_strength = (((item_strength + 1)*2)+1)
	return film_strength


def average_rating(sims, film_id, ratings):
	# counts and totals for each type of aspect
	nRatings = 0
	dRatings = 0

	for sim in sims:
		user_id = sim[0]
		similarity = sim[1]

		if (film_id, user_id) in ratings.keys():
			user_rating = ratings[(film_id, user_id)]
			scaled_rating = ((user_rating - 1) / 2)-1
			dRatings += scaled_rating * similarity
			nRatings += 1

	if nRatings == 0:
		return None
	return dRatings / nRatings