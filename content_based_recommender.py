#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 20:02:11 2018

@author: rushikesh
"""

"""
We will build recommender which recommends movies based on content i.e plots 
of movies. It will recommend movies similar to a specific movie based on the 
pairwise similarity scores.
This is similar to my repo called document_retrieval
"""


import pandas as pd
import numpy as np


#Importing the dataset

""" 
To avoid the following datatype warning specified low_memory=False

DtypeWarning: Columns (10) have mixed types. Specify dtype option on import 
or set low_memory=False.
interactivity=interactivity, compiler=compiler, result=result)

"""
metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

metadata['overview'].head(10)

#This is the data we are targetting

""" Since this data containd plain english text it is not suitable for analysis
purpose. We will use the Tf-Idf vectorizer to convert this text into word vectors.

This will give us a matrix where each column represents a word in the overview 
of the vocabulary which is essentially collection of all the words and each row 
represents a movie.

Tf-Idf is the frequency of words generalized to accomodate the fact that some 
less important words like 'the', 'a' occur too many times across all docs
which might give a fale indicative of the similarity of the docs.

Tf-Idf's are calculated as follows-
     
    #reminder  Put the formula here
    
"""

#Using the sklearns tf-idf vectorizer to create desired word_vectors-

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
#Stop words remove words like 'a', 'the' etc which are not useful for analysis

#Replace Nan in the dataset's overview with empty string-

metadata['overview'] = metadata['overview'].fillna('')

#Constructing a tfidf matrix-
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

#outputting the shape of tfidf_matrix
tfidf_matrix.shape

#We have 45,466 movies and 75827 significant and unique words in our vocab

""" 
We can use this matrix to calculate the similarity between movies.

These are the choices for the distance(which basically acts as a similarity score) -
    Euclidean, pearson, cosine_simlarity
    
You can try diff scores, we will use cosine_similarity for now since it is 
independent of magnitude and is relatively easy and fast to calculate
(especially when used in conjunction with tfidf's)

Mathematically-
    
    cosine(x,y) = [(x).dot(y)]/[||x||.||y||]
    
Since we have used tf-idf vectorizer the resulting vectors for a movie i.e(x and y)
are unit vectors
thus we need to calculate only the dot product of x and y.
We will use slearn's linear_kernel() instead of cosine_similarities() since it 
does the job in this case and is faster.

"""

from sklearn.metrics.pairwise import linear_kernel

#Computing the cosine simiarity-
cos_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

"""
Now we define a function that takes a movie title and returns  list of 10
movies as output which are closest to the given movie.
For this we need a reverse mapping of movie titles and dataframe indices.
We need a mechanism to identify the index of movie from its title.

"""

indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

def get_recommendations(title, cos_sim):
    
    #get the index of movie
    idx = indices[title]
    
    #Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cos_sim[idx]))
    
    #sort the movies based on sim_scores
    sim_scores = sorted(sim_scores, key= lambda x:x[1], reverse=True)
    
    #Get the top 10
    sim_scores = sim_scores[1:11]
    
    #Get the movie_indices
    movie_indices = [i[0] for i in sim_scores]
    
    return metadata['title'].iloc[movie_indices]




get_recommendations('The Dark Knight Rises')








