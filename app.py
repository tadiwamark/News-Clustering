"""
Student Details:

Tadiwanashe Nyaruwata R204445V HAI 
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import os


# Function to load data
def load_data():
    path = '.'  
    frames = []
    for file in os.listdir(path):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(path, file))
            df = df.dropna(subset=['title', 'story', 'url'])
            frames.append(df)
    return pd.concat(frames, ignore_index=True)

# Preprocess and vectorize data
def preprocess_data(data):
    data = data.copy()  
    data['story'] = data['story'].fillna('')  
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['story'])
    return tfidf_matrix


# Clustering function
def apply_clustering(tfidf_matrix, algorithm='Agglomerative', n_clusters=4, linkage='ward'):
    if algorithm == 'Agglomerative':
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        tfidf_matrix = tfidf_matrix.toarray()  # Agglomerative requires dense matrix
        model.fit(tfidf_matrix)
        return model.labels_, model



def main():
    st.title('News Clustering Based on 4 Categories')
    data = load_data()
    if st.checkbox('Show raw data'):
        st.write(data)

    tfidf_matrix = preprocess_data(data)
    linkage_method = st.selectbox('Select Linkage Method', ['ward', 'complete', 'average', 'single'])
    """Complete Linkage is perfoming best with our current data"""
    
    labels, model = apply_clustering(tfidf_matrix, algorithm='Agglomerative', linkage=linkage_method)
    data['cluster'] = labels
    

    selected_cluster = st.selectbox('Select a Cluster', np.unique(data['cluster']))
    filtered_data = data[data['cluster'] == selected_cluster]
    
    if st.checkbox('Show URLs in selected cluster'):
        for url in filtered_data['url']:
            st.write(url)

if __name__ == '__main__':
    main()








    
   




