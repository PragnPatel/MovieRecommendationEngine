import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

data = pd.read_csv('data.csv')

def create_sim():
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    sim = cosine_similarity(count_matrix)
    return data,sim

def recommend(movie):
    movie = movie.lower()
    # check if data and sim are already assigned
    try:
        data.head()
        sim.shape
    except:
        data, sim = create_sim()
    # check if the movie is in our database or not
    if movie not in data['movie_title'].unique():
        return('This movie is not in our database.\nPlease check if you spelled it correct.')
    else:
        # getting the index of the movie in the dataframe
        i = data.loc[data['movie_title']==movie].index[0]

        # fetching the row containing similarity scores of the movie
        # from similarity matrix and enumerate it
        lst = list(enumerate(sim[i]))

        # sorting this list in decreasing order based on the similarity score
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)

        # taking top 1- movie scores
        # not taking the first index since it is the same movie
        lst = lst[1:4]

        # making an empty list that will containg all 10 movie recommendations
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l

st.markdown("Movie Recommendation System")
movie_list = data['movie_title'].values
selected_movie = st.selectbox("Type or select a movie from the dropdown",movie_list)

if st.button("Show Recommendation"):
    recommended_movies = recommend(selected_movie)
    for i in recommended_movies:
        st.subheader(i)