import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('ml-latest-small/ratings.csv')

##print(df.head())

movie_titles = pd.read_csv('ml-latest-small/movies.csv')

##print(movie_titles)

## merge the movie title into the ratings table
df = pd.merge(df, movie_titles, on='movieId')
##print(df.info())

## Group by title and calculate mean rating
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
##print(ratings.head())

## Group by title and count number of ratings per movie
ratings['number_of_ratings'] = df.groupby('title')['rating'].count()
##print(ratings.head())

movie_matrix = df.pivot_table(index='userId', columns='title', values='rating')
##print(movie_matrix.head())

##print(ratings.sort_values('number_of_ratings', ascending=False).head(10))

def recommendmovie(moviename):
    user_rating = movie_matrix[moviename]

    similar = movie_matrix.corrwith(user_rating)

    corr = pd.DataFrame(similar, columns=['correlation'])
    corr.dropna(inplace=True)

    corr = corr.join(ratings['number_of_ratings'])

    recommended_movie = corr[corr['number_of_ratings'] > 100].sort_values(by='correlation', ascending=False).head(20)
    
    print(recommended_movie)
    
while(1):
    print("Type exit/bye/quit or press Enter to Quit")
    moviename = input("Enter movie name with year in braces to get suggestions: ")
    if(moviename=="" or moviename=="bye" or moviename=="quit" or moviename=="exit"): break
    print("Building Recommendation Model...............")
    recommendmovie(moviename)
    print("=======================================================================================")
    
