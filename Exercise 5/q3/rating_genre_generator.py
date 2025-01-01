#!/usr/bin/python3
import pandas as pd

# Following snippet of code is being used to generate a new .dat file which in simple words contain 
# a join of data from movies.dat and ratings.dat 
movies_data = pd.read_csv('movies.dat', sep = '::', names=['Movie_Name', 'Genres'], engine = 'python')
ratings_data = pd.read_csv('ratings.dat', names = ['user_id', 'movie_id', 'rating', 'time'], sep = "::", engine = 'python')
rating_genre_data = ratings_data.merge(movies_data['Genres'], left_on = 'movie_id', right_index = True)
rating_genre_data['Genre'] = rating_genre_data.Genres.apply(lambda x:x.replace("|", ":"))
rating_genre_data.drop('Genres', axis = 1, inplace = True)
rating_genre_data.to_csv('ratings_genre.dat', sep = '|', index = False, header = False)