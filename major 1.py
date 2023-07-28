#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import ipywidgets as widgets
from IPython.display import display


# In[14]:


ratings=pd.read_csv(r"C:\Users\91830\Downloads\ratings.csv")
movies=pd.read_csv(r)


# #EXPLORATORY DATA ANALYSIS

# In[15]:


ratings.head()


# In[16]:


movies.head()


# In[17]:


movies.shape


# In[18]:


ratings.shape


# In[19]:


ratings.info()


# In[20]:


movies.info()


# In[21]:


movies.nunique()


# In[22]:


movies.isnull().sum()


# In[23]:


ratings.isnull().sum()


# In[24]:


movie_genre_ratings = movies.merge(ratings, on='movieId')
average_genre_ratings = movie_genre_ratings.groupby('genres')['rating'].mean()
total_movies_genre = movie_genre_ratings['genres'].value_counts()
print("AVERAGE RATING BY GENRE: ")
print(average_genre_ratings)
print("\nTOTAL MOVIE BY GENRE :")
print(total_movies_genre)


# # TASK 1 

# #Create a popularity-basedrecommender system at a genre level

# In[26]:


ratings=pd.read_csv(r"C:\Users\91830\Downloads\ratings.csv")
movies=pd.read_csv(r"C:\Users\91830\Desktop\nf2\movies.csv")


# In[27]:


genre = input('Enter the genre: ')
genre_movies = movies[movies['genres'].str.contains(genre)]

threshold = int(input("Enter the minimum review threshold: "))
review_counts = ratings['movieId'].value_counts().rename('review_count')
genre_movies = genre_movies.merge(review_counts, left_on='movieId', right_index=True)
genre_movies = genre_movies[genre_movies['review_count'] >= threshold]

average_ratings = ratings.groupby('movieId')['rating'].mean().rename('average_rating')
genre_movies = genre_movies.merge(average_ratings, left_on='movieId', right_index=True)
movies_sorted = genre_movies.sort_values(by='average_rating', ascending=False)

N = int(input("Enter the number of recommendations: "))
recommended_movies = movies_sorted.head(N)

print(recommended_movies[['movieId', 'title', 'average_rating']])
pd.DataFrame(recommended_movies)


# # TASK 2

# #Create a content-based recommender system

# In[28]:


ratings=pd.read_csv(r"C:\Users\91830\Downloads\ratings.csv")
movies=pd.read_csv(r"C:\Users\91830\Desktop\nf2\movies.csv")


# In[31]:


movie_title = input('Enter the Movie Title: ')

N = int(input('Enter number of recommendations: '))

if movie_title not in movies['title'].values:
    print("Movie title not found in the dataset.")
else:
    selected_movie = movies[movies['title'] == movie_title].iloc[0]

    selected_movie_genres = selected_movie['genres']

    similar_movies = movies[movies['genres'].apply(lambda x: any(genre in x for genre in selected_movie_genres))]

    if len(similar_movies) > 0:
        similarity_scores = similar_movies.apply(lambda row: sum(genre in selected_movie_genres for genre in row['genres']), axis=1)
        similar_movies.loc[:, 'similarity_score'] = similarity_scores

        sorted_movies = similar_movies.sort_values(by='similarity_score', ascending=False)

        recommended_movies = sorted_movies.head(N)

        recommended_movies = pd.DataFrame(recommended_movies[['movieId', 'title', 'similarity_score']])
        print(recommended_movies)
    else:
        print("No similar movies found.")
pd.DataFrame(recommended_movies)


# # TASK 3

# #Create a collaborative based recommender system

# In[32]:


ratings=pd.read_csv(r"C:\Users\91830\Downloads\ratings.csv")
movies=pd.read_csv(r"C:\Users\91830\Desktop\nf2\movies.csv")


# In[33]:


ratings.head()


# In[35]:


user_id = int(input('UserID: '))
N = int(input('Num Recommendations: '))
K = int(input('Threshold: '))

similarity_matrix = cosine_similarity(ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0))
similar_users_indices = similarity_matrix[user_id-1].argsort()[::-1][1:K+1]

movies_rated_by_similar_users = ratings[ratings['userId'].isin(similar_users_indices + 1)]
average_ratings = movies_rated_by_similar_users.groupby('movieId')['rating'].mean().rename('average_rating')

sorted_movies = average_ratings.sort_values(ascending=False)
recommended_movies = sorted_movies.head(N)

pd.DataFrame(recommended_movies)


# In[ ]:




